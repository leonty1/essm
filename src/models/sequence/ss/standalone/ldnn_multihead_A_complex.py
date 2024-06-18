""" Standalone version of MIMO SSM with general architecture.
Multi-head setting
A B C are complex numbers, D and delta not
bidirectional kernel without introducing addtional parameters
"""

import logging
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from src.models.nn.components import LinearActivation, get_initializer, Activation

import src.utils as utils
from src.models.sequence.pool import registry as pool_registry
from src.models.sequence.pool import DownSample_ms, UpSample_ms

from einops import rearrange, repeat
import opt_einsum as oe

from scipy import signal

einsum = contract = oe.contract
contract_expression = oe.contract_expression

from omegaconf import DictConfig

import sys

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

log = get_logger(__name__)


""" Optimizer utilities """

class OptimModule(nn.Module):
    """
    Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters
    """

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter
        args
            name: name of the buffer/parameter
            tensor: tensor to register
            trainable: whether to register as a trainable parameter (default: False)
            lr: learning rate to use for this parameter (default: None)
            wd: weight decay to use for this parameter (default: None)
        """

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


""" Misc functional utilities """

_c2r = torch.view_as_real       # complex to real，a+bi --> [a,b] -
_r2c = torch.view_as_complex        # real to complex， [a,b] --> a+bi

def reciprocal(x, epsilon=1e-7, clamp=False):
    """ input real or complex number x, returns 1 / x, with bounded norm """
    x_conj = x.conj()
    norm_sq = (x*x_conj).real.clamp(epsilon) if clamp else (x*x_conj + epsilon)
    return x_conj / norm_sq

def block_diag_tensor(input):
    '''
    merge multi-head parameters. B C D
    input [channel head N M]
    out  [channel head*N head*M] block-diagonal tensor
    '''
    # C, H, N, M=input.shape
    # output_tensor = torch.zeros(C, H*N, H*M)
    output_list = [torch.block_diag(*input[i, ...].unbind(dim=0)) for i in range(input.shape[0])]
    block_diag_tensor = torch.stack(output_list, dim=0)
    return block_diag_tensor

##############SSM Initialization functions#########
#Hippo for initialization A B C
def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix, resued from S5.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = torch.sqrt(1 + 2 * torch.arange(N, device='cpu'))       #[N]
    A = P.unsqueeze(1) * P.unsqueeze(0)     #[NN]
    out = torch.tril(A) - torch.diag(torch.arange(N, device='cpu'))          # [N N] real number
    return -out

def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)       #[N N] real number

    # Add in a rank 1 term. Makes it Normal.
    P = torch.sqrt(torch.arange(N, device='cpu') + 0.5)     #[N] real number

    # HiPPO also specifies the B matrix
    B = torch.sqrt(2 * torch.arange(N, device='cpu') + 1.0)         #[N] real number
    return hippo, P, B

def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda-[N], low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)  # A [N N], P [N], B [N]
    #
    A = A.type(torch.complex64)
    P = P.type(torch.complex64)
    B = B.type(torch.complex64)

    S = A + P.unsqueeze(1) * P.unsqueeze(0)  # [N N]
    # S = S.type(torch.complex64).cuda()

    S_diag = torch.diagonal(S)  # [N]
    Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag, device='cpu')  ##[N N]

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = torch.linalg.eig(S * -1j)  # Lambda_imag-[N], V-[N N] complex

    P = V.conj().T @ P  # [N] complex
    B_orig = B  # [N] real number
    B_out = V.conj().T @ B  # initial value for B complex

    return Lambda_real + 1j * Lambda_imag, P, B_out, V, B_orig

def ssm_initializer(channel,head,d_input,d_state,d_output, Lambda_init, B_init, C_init, conj_sym,keep_d_state):
    '''
    return them all in real representations
    complex
    Lambda, [Channel, d_state, 2]
    B, [Channel, d_state, d_input, 2]
    C, [Channel, d_output, d_state, 2]
    or real
    Lambda, [Channel, N]
    B, [Channel, d_state, d_input]
    C, [Channel, d_output, d_state]
    '''
    initializer_list = ['uniform', 'kaiming_uniform', 'xavier_uniform', 'normal', 'kaiming_normal', 'xavier_normal', 'trunc_normal', 'zero', 'one', 'half']

    if Lambda_init == 'hippo':
        if conj_sym and keep_d_state:
            # double the d_state to keel size unchange -- reference S4D in https://github.com/state-spaces/s4
            d_state = d_state*2

        Lambda0, _, B, V, B_orig = make_DPLR_HiPPO(d_state)

        if conj_sym:
            # Only keep one of the conjugate pairs, only used for following lambada and V
            d_state = d_state // 2

        Lambda = Lambda0[:d_state]  # Block-diagonal in S5 is equivalent to our multi-head setting
        Lambda=Lambda.unsqueeze(0).expand(channel, head, -1).to(torch.cfloat)#.clone()   #copy for channels [channel, d_state]
        Lambda = _c2r(Lambda)

        V = V[:, :d_state]   # if conj_sym [d_state, d_state/2], else [d_state, d_state]
        Vinv = V.conj().T # if conj_sym [d_state/2, d_state], else [d_state, d_state]

        if conj_sym:
            d_state = d_state*2  #restore the size

        #B hippo initialization
        B_shape = (d_state, d_input)
        B_real = torch.empty(B_shape, dtype=torch.float, device='cpu')
        # torch.nn.init.kaiming_uniform_(B_real, a=0)
        # nn.init.constant_(B_real, 1)
        # nn.init.normal_(B_real)
        nn.init.trunc_normal_(B_real)
        B_comp = torch.empty(B_shape, dtype=torch.float, device='cpu')
        # torch.nn.init.kaiming_uniform_(B_comp, a=0)
        # torch.nn.init.constant_(B_comp, 1)
        # nn.init.normal_(B_comp)
        nn.init.trunc_normal_(B_comp)
        B_0 = torch.complex(B_real, B_comp)
        B_v = Vinv @ B_0

        #C hippo initialization
        C_real= torch.empty(channel, head, d_output, d_state, device='cpu')  # [Channel, d_output, d_state]
        C_comp= torch.empty(channel, head, d_output, d_state, device='cpu')  # [Channel, d_output, d_state]
        # torch.nn.init.trunc_normal_(C_real, mean=0.0, std=1.0, a=-2, b=2.0)
        # torch.nn.init.trunc_normal_(C_comp, mean=0.0, std=1.0, a=-2, b=2.0)
        # torch.nn.init.normal_(C_real)
        torch.nn.init.trunc_normal_(C_real)
        # torch.nn.init.normal_(C_comp)
        torch.nn.init.trunc_normal_(C_comp)
        C_rand=torch.complex(C_real, C_comp).to(torch.cfloat)

        # C_complex = torch.einsum('chmn,nd->chmd', C_rand, V)        #[C head M N]
        C_complex = contract('chmn,nd->chmd', C_rand, V)

    elif Lambda_init in initializer_list:
        Lambda_initializer=get_initializer(Lambda_init)
        Lambda = torch.empty(channel, head, d_state, 2, dtype=torch.float, device='cpu')
        Lambda_initializer(Lambda)
    else:
        raise ValueError(f"Lambda init {Lambda_init} is not implemented")

    if B_init == 'hippo' and Lambda_init == 'hippo':
        B = B_v.unsqueeze(0).expand(channel, head, -1, -1).cuda()#.type(torch.cfloat)       #copy for channels  [C head d_state, d_input]
        B = _c2r(B)
    elif B_init in initializer_list:
        B_initializer = get_initializer(B_init)
        B = torch.empty(channel, head, Lambda.shape[-2], d_input, 2, dtype=torch.float, device='cpu')
        B_initializer(B)
    else:
        raise ValueError(f"B init {B_init} is not implemented")

    if C_init == 'hippo' and Lambda_init == 'hippo':
        C = _c2r(C_complex)
    elif C_init in initializer_list:
        C_initializer = get_initializer(C_init)
        C = torch.empty(channel, head, d_output, Lambda.shape[-2], 2, dtype=torch.float, device='cpu') # [Channel, head, d_output, d_state, 2]
        C_initializer(C)
    else:
        raise ValueError(f"C init {C_init} is not implemented")

    assert Lambda.dim()==4
    assert B.dim()==5
    assert C.dim()==5

    return -torch.abs(Lambda), B , C

# initialize log_dt
def log_dt_initializer(channel, head, conj_sym,keep_d_state, Lambda_init, d_dt,dt_max=0.1,dt_min=0.001,d_dt_is_n=True):
    '''
    Initialize the learnable timescale Delta by sampling uniformly between dt_min and dt_max. ref S5, d_dt=N may better than d_dt=1
     Args:
         dt_min (float64): minimum value
         dt_max (float64): maximum value
    return real [Channel, head, N]
    '''

    if conj_sym and Lambda_init == 'hippo':
        d_dt=d_dt//2
        if keep_d_state:
            d_dt = d_dt * 2

    # dt_init = torch.empty(channel, head, d_dt)
    if d_dt_is_n:
        dt_init = torch.empty(channel, head, d_dt)
    else:
        dt_init = torch.empty(channel, head, 1, device='cpu')
    torch.nn.init.uniform_(dt_init, a=0.0, b=1.0)
    log_dt = torch.log(torch.tensor(dt_min)) + dt_init * (
            torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))  # [Channel, head, d_dt] - real

    return log_dt.cuda()

#######################constrain A with negtive valued real part in continuous SSM
def neg_real_Lambda(version, Lambda, max_real_Lambda):
    if 'softplus' in version:
        Neg_real_lambda =  -F.softplus(Lambda[...,0])  + 1j*Lambda[...,1]
    elif 'sigmoid' in version:
        Neg_real_lambda =   -F.sigmoid(Lambda[...,0])  + 1j*Lambda[...,1]
    elif 'Gaussian' in version:
        Neg_real_lambda =   -torch.exp(-Lambda[...,0]**2)  + 1j*Lambda[...,1]
    elif 'clip' in version:
        Neg_real_lambda =   Lambda[...,0].clip(max=-max_real_Lambda) + 1j*Lambda[...,1]
    else:
        raise NotImplementedError(f"version {version} is not implemented")
    return Neg_real_lambda

def imag_constrain(value, version, max_value=1):
    #['softplus', 'sigmoid', 'Gaussian',  'clip', 'zero', 'negtive',['identity', None, 'same']]
    if 'softplus' in version:
        constrained_value = value[...,0]  + 1j*F.softplus(value[...,1])         #(0, ~)
    elif 'sigmoid' in version:
        constrained_value =   value[...,0]  + 1j*F.sigmoid(value[...,1])        #(0, 1)
    elif 'Gaussian' in version:
        constrained_value =   value[...,0]  + 1j*torch.exp(value[...,1]**2)     #(0, 1)
    elif 'clip' in version:
        constrained_value =   value[...,0] + 1j*torch.clip(value[...,1], max=max_value, min=-max_value)     #(-1, 1)
    elif 'zero' in version:
        constrained_value = value[..., 0] + 1j * (value[..., 1]*0)      # 0
    elif version in ['identity', None, 'same']:
        constrained_value = value[..., 0] + 1j * value[..., 1]          # (~, ~)
    elif 'negtive' in version:
        constrained_value = value[..., 0] + 1j*torch.clip(value[...,1], max=-0.0001)  #(~, 0)
    else:
        raise NotImplementedError(f"constrain version {version} is not implemented")
    return constrained_value

######################## Discretization functions
def discretize_GBT(Lambda, B, Delta, alpha=0.5):
    """ Discretize a diagonalized, continuous-time linear SSM
        using generalized bilinear transform (GBT)  method.
        alpha=0 is forward Euler,
        alpha=0.5 is bilinear transform,(also known as Tustin's method or trapezoidal rule )
        alpha=1 is backward Euler,

        Args:
            Lambda (complex64): diagonal state matrix              complex  or  real  [Channel, d_state]
            B (complex64): input matrix                                      complex  or  real  [Channel, d_state, d_input]
            Delta (float64): discretization step sizes                  complex  or  real [Channel, d_state]
        Returns:
            discretized Lambda_bar           [Channel, d_state]
            B_bar (complex64)                   [Channel, d_state]
    """
    Identity = torch.ones(Lambda)                   #[Channel, d_state]
    BL = 1 / (Identity - Delta*alpha* Lambda)   #[Channel, d_state]
    Lambda_bar = BL * (Identity + Delta*(1-alpha)* Lambda)   #[Channel, d_state]
    delta_B = torch.einsum('cn,cnh->cnh', Delta, B)                 #[Channel, d_state, d_input]
    B_bar=torch.einsum('cn,cnh->cnh', BL, delta_B)                #[Channel, d_state, d_input]
    return Lambda_bar, B_bar

def discretize_zoh(Lambda, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix            complex  or  real  [Channel, head*d_state]
            B  complex  or  real  [Channel, d_state, d_input]
            Delta (float64): discretization step sizes            complex  or  real [Channel, head*d_state]
        Returns:
            Lambda_bar complex64 or float64 [Channel, d_state]
            B_bar complex64 or float64 [Channel, d_state]
    """
    Lambda_bar = Lambda * Delta   #  A*delta  -  [Channel, head, d_state] complex or real
    Identity = torch.ones_like(Lambda).cuda()       #[Channel, head, d_state]
    B_coef = (reciprocal(Lambda) * (torch.exp(Lambda_bar)-Identity))   # [C S N]

    return Lambda_bar, B_coef

def discrete_ssm(Lambda, log_dt, discrete_method):
    """ Discretize a diagonalized, continuous-time linear SSM.
            Args:
                Lambda (complex64): diagonal state matrix              [Channel, d_state]
                B   [Channel, d_state, d_input]
                Delta (float64): discretization step sizes            complex log_dt or  real [Channel, d_state]
            Returns:
                discretized Lambda_bar complex64 or float64 [Channel, d_state]
                B_bar float64 [Channel, d_state, d_input]
        """

    Delta=log_dt.exp()      #complex [Channel, d_state]
    # Delta=0.0625*Delta

    if discrete_method == 'zoh':
        Lambda_bar, B_coef =discretize_zoh(Lambda, Delta)
    elif discrete_method == 'forward_euler':
        Lambda_bar, B_coef = discretize_GBT(Lambda,Delta, alpha=0.)
    elif discrete_method == 'bilinear':
        Lambda_bar, B_coef = discretize_GBT(Lambda, Delta, alpha=0.5)
    elif discrete_method == 'backward_euler':
        Lambda_bar, B_coef = discretize_GBT(Lambda, Delta, alpha=1.)
    else:
        raise NotImplementedError(f"discrete method {discrete_method} is not implemented")

    return Lambda_bar, B_coef


######### SSM layer
class SSM(OptimModule):
    '''
    SSM
    input [B, H, L]，
    output [B, M, L]，
    state[B, N, L]
    '''
    def __init__(
            self,
            d_input=1,                   # dimension of the input. 面向向量的建模，维度是d_input
            d_output=None,               # dimension of the output
            #d_m,
            d_s=256,         # dimension of the state
            bidirectional=False,
            channel=1,
            head=1,
            dropout=0.,
            dt_min=1e-3,
            dt_max=1e-1,
            d_dt_is_n=True,
            trainable=None,
            lr=None,
            Lambda_init='hippo',
            max_real_Lambda=1e-4,
            B_init ='one',
            C_init='one',
            D_init='one',
            D_value=0,
            l_max=1,               #length of data in original dataset
            transposed=True,
            complex_ssm=True,  # use separate deltas for real, imag parts of Lambda
            conj_sym=True,
            keep_d_state=False,
            # SSM Kernel arguments
            discrete_method = 'zoh',  # 'options: ['zoh': zero-order hold method, 'bilinear': bilinear transform, 'forward_euler': forward euler method, 'backward_euler': backward euler method]'
            version_A_neg='clip', # method to ensure negtive real part of Lambda
            version_A_imag='same',
            version_B_imag='same',
            version_C_imag='same',
            max_kernel_length=None,  # max len of SSM kernel to be used, only used for specific setting
            activation='gelu',  # activation in between SS and Mixer
            acttivation_post=None,  # activation after Mixer
            **kernel_args,
        ):
        """
        State Space model with multi-input and multi-output as a basic Layer (SSM)
        d_input: the dimension of the input, also denoted by H
        d_output: the dimension of the output, also denoted by M
        d_state: the dimension of the state, also denoted by N,     Note: N denotes half the true state size, because of conjugate symmetry
        keep_d_state: if True N=2N, d_state is the true state size; if false, N denotes half the true state size
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of ssm in parallel connection
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]
        """

        super().__init__()

        self.h = d_input//head
        self.n = d_s//head      #directly give d_state in each head, not d_s//head
        if d_output is None:
            d_output = d_input      #input output have same size
        self.m = d_output // head
        self.channel=channel
        self.head=head
        self.bidirectional = bidirectional
        self.transposed = transposed
        self.complex_ssm = complex_ssm
        self.discrete_method=discrete_method
        self.dt_min=dt_min
        self.dt_max=dt_max
        self.d_dt_is_n = d_dt_is_n
        self.D_value = D_value
        self.max_real_Lambda=max_real_Lambda
        self.conj_sym=conj_sym
        self.max_kernel_length=max_kernel_length
        self.version_A_neg=version_A_neg
        self.version_A_imag = version_A_imag
        self.version_B_imag = version_B_imag    # not used in thisversion
        self.version_C_imag = version_C_imag    # not used in thisversion
        self.activation = activation
        self.acttivation_post=acttivation_post

        #Dropout
        # dropout_seq = nn.Dropout1d if self.transposed else nn.Dropout
        dropout_seq = nn.Dropout if self.transposed else nn.Dropout
        dropout_element = nn.Dropout
        self.dropout_seq = dropout_seq(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout_element = dropout_element(dropout) if dropout > 0.0 else nn.Identity()

        ######## continuous ssm ############
        Lambda, B, C= ssm_initializer(channel=self.channel,head=self.head, d_input=self.h,d_state=self.n,d_output=self.m,Lambda_init=Lambda_init, B_init=B_init, C_init=C_init,conj_sym=self.conj_sym,keep_d_state=keep_d_state)
        B=B[...,0]
        C=C[...,0]
        log_dt=log_dt_initializer(channel=self.channel, head=self.head, d_dt=self.n, dt_max=self.dt_max, dt_min=self.dt_min, d_dt_is_n=self.d_dt_is_n,conj_sym=self.conj_sym,Lambda_init=Lambda_init,keep_d_state=keep_d_state)  #[C head n]

        # set Lambda, B, log_dt learnable with given learning rate
        self.lr = DictConfig(
            {"log_dt": 1e-3, "Lambda": 1e-3, "B": 1e-3})  # learning rates for Lambda, B, log_dt
        if lr is not None:
            self.lr.update(lr)

        self.trainable = DictConfig({"log_dt": True, "Lambda": True, "B": True, "C": True, "D": True})  # which parameters to train.
        if trainable is not None:
            self.trainable.update(trainable)

        self.register("log_dt", log_dt.cuda(), self.trainable.log_dt, self.lr.log_dt, wd=0.0)  # [channel head N] or [channel head N 2]
        self.register("Lambda", Lambda.cuda(), self.trainable.Lambda, self.lr.Lambda, wd=0.0)  # [channel head N] or [channel head N 2]
        self.register("B", B.cuda(), self.trainable.B, self.lr.B, wd=0.0)  # [Channel head d_state d_input]  or [Channel head d_state d_input 2]
        self.register("C", C.cuda(), self.trainable.C, self.lr.C, wd=0.0)

        # Initlalize coefficients D - set as real numbers, no need to be complex numnbers  - [C head M H]
        D_initializer = get_initializer(D_init)
        if D_value == 0:
            self.D = torch.zeros(channel, self.head, self.m, self.h, device='cuda')
        elif D_value == 1 and self.m == self.h:
            # D is identity matrix
            # self.D = torch.eye(self.m, device='cuda').unsqueeze(0).expand(self.channel, self.head, -1, -1)  # can be seen as skip connection
            D = torch.empty(channel, self.head, self.m, device='cuda')  # [C, S, M]
            D_initializer(D)
            self.register("D", D.cuda(), self.trainable.D, self.lr.D, wd=0.0)
        else:
            D = torch.empty(channel, self.head, self.m, self.h, device='cuda')  # [C, S, M, H]  # can be seen as skip connection with affine function
            D_initializer(D)
            self.register("D", D.cuda(), self.trainable.D, self.lr.D, wd=0.0)

        # linear layer to head and mix channels
        self.use_head_mixer = False
        if self.head > 1:
            self.use_head_mixer = True
            self.head_mixer = nn.Linear(self.m * self.head, self.m * self.head)

        self.use_channel_mixer = False
        if channel > 1:
            self.use_channel_mixer = True
            self.channel_mixer = nn.Linear(self.m *self.head* channel, d_output)

        # activation - before head mixer
        if self.activation in ["full_glu"]:
            self.out1 = nn.Linear(self.m*self.head, self.m*self.head)
            self.out2 =nn.Linear(self.m*self.head, self.m*self.head)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Linear(self.m*self.head, self.m*self.head)
        else:
            self.activation_func = Activation(self.activation)

        # activation post - after channel mixer
        if self.acttivation_post in ["full_glu"]:
            self.out1_post= nn.Linear(d_output, d_output,)
            self.out2_post = nn.Linear(d_output, d_output)
        elif self.acttivation_post in ["half_glu1", "half_glu2"]:
            self.out2_post = nn.Linear(d_output, d_output)
        else:
            self.activation_post = Activation(self.acttivation_post)

    def forward(self, u, **kwargs):

        if not self.transposed: u = u.transpose(-1, -2)
        batch, d_in, length = u.shape
        u = u.reshape(batch, self.head, self.h, length)     #[B head H L]

        u_D=u

        # constrain A with negtive valued real part in continuous SSM  - [channel head N]
        Lambda = neg_real_Lambda(version=self.version_A_neg, Lambda=self.Lambda,max_real_Lambda=self.max_real_Lambda)  # returen real or complex

        # constrain the imaginary part of A B C      ['softplus', 'sigmoid', 'Gaussian',  'clip', 'zero', ['identity', None, 'same']]
        Lambda = _c2r(Lambda)
        Lambda = imag_constrain(value=Lambda, version=self.version_A_imag)

        ######Discrete SSM -------------------  dicretize A B for forward inference
        if not self.d_dt_is_n:
            log_dt=self.log_dt.expand(-1,-1, self.n)    #c s n
        else:
            log_dt = self.log_dt
        assert Lambda.shape == log_dt.shape
        Lambda_bar, B_coef= discrete_ssm(Lambda=Lambda, log_dt=log_dt,
                                         discrete_method=self.discrete_method)
        B_u = contract('csnh,bshl->bcsnl', self.B, u)
        L = u.size(-1)

        ############## Covolutional SSM
        # Compute SS Kernel
        Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)

        if self.discrete_method == 'zoh':
            length = torch.arange(Lk).cuda()
            p = Lambda_bar.unsqueeze(-1) * length
            state_kernel = torch.einsum('csnl,csn->csnl', p.exp(), B_coef)
        else:
            raise NotImplementedError(f"discrete method {self.discrete_method} is not implemented")

        #bidirectional kernel for non-causal state inference
        state_kernel = state_kernel.real
        if self.bidirectional:
            # k0, k1 = rearrange(state_kernel, '(c s) n l -> c s n l', s=2)
            # k0, k1 = torch.chunk(state_kernel, 2, dim=0)
            state_kernel_new = F.pad(state_kernel, (0, L)) + F.pad(state_kernel.flip(-1), ( L, 0))  # bidirectional without additional parameters
        else:
            state_kernel_new = state_kernel

        ############## Convolution for state  inference       # fast implementation via FFT
        Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)
        n = L + Lk
        k_f = torch.fft.rfft(state_kernel_new, n=n)  # [channel, S, N, L]  L~n/2    [4, 512, 1025]  torch.cfloat    torch.Size([128, 2048])
        u_f = torch.fft.rfft(B_u, n=n)  # [B channel, N, L]  L~n/2)                    [4, 4, 1025])   torch.cfloat   torch.Size([100, 128, 2048])
        # x_f = torch.einsum('bcsnl,csnl->bcsnl', u_f, k_f)      #[B channel, N, L]
        x_f = contract('csnl,bcsnl->bcsnl', k_f, u_f)
        x = torch.fft.irfft(x_f, n=n)[..., :L]                           # [B channel, N, L]

        C_x = contract('csmn, bcsnl->bcsml', self.C, x)
        if self.conj_sym:
            C_x = 2*C_x

        #output function  -- also as spatial reduction/reconstruction
        if self.D_value==0:
            y = C_x
        elif self.D_value == 1 and self.m == self.h:
            # y = C_x+ torch.einsum('csm, bsml->bcsml', self.D, u_D)
            y = C_x + contract('csm, bsml->bcsml', self.D, u_D)
        else:
            # y = C_x + torch.einsum('csmh, bshl->bcsml', self.D, u_D)
            y = C_x + contract('csmh, bshl->bcsml', self.D, u_D)

        batch = int(y.size(0))
        channel = int(y.size(1))
        y = rearrange(y, 'b c s m l -> (b c) l (s m)')

        # activation - before head/channel mixers  like DSS
        if self.activation in ["full_glu"]:
            y = self.dropout_seq(F.gelu(y))
            y = self.out1(y) * F.sigmoid(self.out2(y))

        elif self.activation in ["half_glu1"]:
            y = self.dropout_seq(F.gelu(y))
            y = y * F.sigmoid(self.out2(y))

        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            y1 = self.dropout_seq(F.gelu(y))
            y = y * F.sigmoid(self.out2(y1))
        else:
            y = self.activation_func(y)

        # head Mixer
        if self.use_head_mixer:
            y = self.dropout_seq(y)
            y = self.head_mixer(y)  # linear to mix features among channels and heads      [B M L]

        #channel mixer  --  not used currently, leave for future possible use
        y = rearrange(y, '(b c) l m -> b l (c m)', b=batch, c=channel)
        if self.use_channel_mixer:
            # y=self.dropout_seq(y)
            y = self.channel_mixer(y)  # linear to mix features among channels and heads      [B M L]

        # Post activation - after head/channel mixers  like S4
        if self.acttivation_post in ["full_glu"]:
            y = self.dropout_seq(F.gelu(y))
            y = self.out1_post(y) * F.sigmoid(self.out2_post(y))

        elif self.acttivation_post in ["half_glu1"]:
            y = self.dropout_seq(F.gelu(y))
            y = y * F.sigmoid(self.out2_post(y))

        elif self.acttivation_post in ["half_glu2"]:
            # Only apply GELU to the gate input
            y1 = self.dropout_seq(F.gelu(y))
            y = y * F.sigmoid(self.out2_post(y1))
        else:
            y = self.activation_post(y)

        assert y.dim()==3

        y = rearrange(y, 'b m l -> b l m')

        return y, None

    @property
    def d_state(self):
        return self.n * self.head

    @property
    def d_output(self):
        return self.m* self.head
