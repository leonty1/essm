""" Implements downsampling and upsampling on sequences """

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from src.models.sequence import SequenceModule
from src.models.nn import LinearActivation

from scipy import signal
""" Simple pooling functions that just downsample or repeat

pool: Subsample on the layer dimension
expand: Repeat on the feature dimension
"""

def downsample(x, pool=1, expand=1, transposed=False):
    if x is None: return None
    if pool > 1:
        if transposed:
            x = x[..., 0::pool]
        else:
            x = x[..., 0::pool, :]

    if expand > 1:
        if transposed:
            x = repeat(x, '... d l -> ... (d e) l', e=expand)
        else:
            x = repeat(x, '... d -> ... (d e)', e=expand)

    return x

def upsample(x, pool=1, expand=1, transposed=False):
    if x is None: return None
    if expand > 1:
        if transposed:
            x = reduce(x, '... (d e) l -> ... d l', 'mean', e=expand)
        else:
            x = reduce(x, '... (d e) -> ... d', 'mean', e=expand)
    if pool > 1:
        if transposed:
            x = repeat(x, '... l -> ... (l e)', e=pool)
        else:
            x = repeat(x, '... l d -> ... (l e) d', e=pool)
    return x


#basic class for pooling time axis for paper 3
def downsample_ms(x, pool=1, transposed=True):
    if x is None: return None
    if pool > 1:
        if transposed:
            x = x[..., 0::pool]
        else:
            x = x[..., 0::pool, :]
    return x

#upsampling use interpolation for paper 3
def upsample_interp(x,up=1, transposed=False, mode='nearst'):
    if x is None: return None
    if up > 1:
        if transposed:
            #interpolate x with scale_facter=pool in last dimention
            x = F.interpolate(x, size= x.shape[-1]*up, mode=mode)
        else:
            #interpolate x with scale_facter=pool in second dimention
            x=x.transpose(-1, -2)
            x = F.interpolate(x, size= x.shape[-1]*up, mode=mode)
            x = x.transpose(-1, -2)
    else:
        pass
    return x


class DownSample(SequenceModule):
    def __init__(self, d_input, pool=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.pool = pool
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        return downsample(x, self.pool, self.expand, self.transposed)

    def step(self, x, state, **kwargs):
        if self.pool > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand

#for paper 3
class DownSample_ms(nn.Module):
    def __init__(self, d_input, down=1,  transposed=True, filtering=False,filter_length=5, cutoff=0.9, kernel_learn=False):
        super().__init__()
        self.d_input = d_input
        self.down=down
        self.transposed = transposed
        self.filtering=filtering
        self.kernel_learn = kernel_learn

        if self.filtering:
            kernel_given = torch.from_numpy(signal.firwin(numtaps=filter_length, cutoff=cutoff)).cuda()
            kernel_channels = repeat(kernel_given, ' k -> d k', d=self.d_input)  # repeat for each channel
            if not self.kernel_learn:
                self.filter_kernel = kernel_channels
            else:
                # self.conv1d = nn.Conv1d(self.d_input, self.d_input, kernel_size=5, bias=False, padding='same',groups=self.d_input)  #为了可以使用指定的初始化，直接使用卷积函数
                # initialize using calcualted kernel
                self.filter_kernel = nn.Parameter(kernel_channels.clone(),
                                                  requires_grad=True)  # learn filter for each channel

                # initialize using given distribution
                # self.filter_kernel= nn.Parameter(torch.empty(self.d_input, filter_length, dtype=torch.float, device='cuda'),  requires_grad=True)
                # nn.init.kaiming_uniform_(self.filter_kernel)
                # torch.nn.init.ones_(self.filter_kernel)
                # nn.init.normal_(self.filter_kernel)

    def forward(self, x):
        if not self.transposed: x = rearrange(x, 'b l d -> b d l')

        if self.filtering:
            # x=self.conv1d(x)        #learn kernel to filter the data
            kernel = repeat(self.filter_kernel, ' d k -> d m k', m=1).float()
            x = F.conv1d(x, kernel, bias=None, padding='same', groups=x.shape[1])

        x=downsample_ms(x, self.down, self.transposed)

        if not self.transposed: x = rearrange(x, 'b d l -> b l d')

        return x

    def step(self, x, state, **kwargs):
        if self.pool > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input

class UpSample(nn.Module):
    def __init__(self, d_input, pool=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.pool = pool
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        return upsample(x, self.pool, self.expand, self.transposed)

    @property
    def d_output(self):
        return self.d_input // self.expand
    def step(self, x, state, **kwargs):
        if self.pool > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

#for paper 3
class UpSample_ms(nn.Module):
    def __init__(self, d_input, up=1,  transposed=True, filtering=False,filter_length=5, cutoff=0.9, kernel_learn=False, mode='linear'):
        super().__init__()
        self.d_input = d_input
        self.up = up
        self.transposed = transposed
        self.filtering = filtering
        self.kernel_learn = kernel_learn

        if self.filtering:
            kernel_given = torch.from_numpy(signal.firwin(numtaps=filter_length, cutoff=cutoff)).cuda()
            kernel_channels= repeat(kernel_given, ' k -> d k', d=self.d_input)    #repeat for each channel
            if not self.kernel_learn: self.filter_kernel = kernel_channels
            else:
                #self.conv1d = nn.Conv1d(self.d_input, self.d_input, kernel_size=5, bias=False, padding='same',groups=self.d_input)  #为了可以使用指定的初始化，直接使用卷积函数
                #initialize using calcualted kernel
                self.filter_kernel= nn.Parameter(kernel_channels.clone(),  requires_grad=True)   #learn filter for each channel
                #initialize using given distribution  - ones/normal/kmuniform
                #self.filter_kernel= nn.Parameter(torch.empty(self.d_input, filter_length, dtype=torch.float, device='cuda'),  requires_grad=True)
                #nn.init.ones_(self.filter_kernel)
                #nn.init.normal_(self.filter_kernel)
                nn.init.kaiming_uniform_(self.filter_kernel)


        self.mode=mode

    def forward(self, x):
        x=upsample_interp(x, up=self.up, transposed=self.transposed, mode=self.mode)
        if not self.transposed: x = rearrange(x, 'b l d -> b d l')

        if self.filtering:
            #x=self.conv1d(x)        #learn kernel to filter the data
            kernel = repeat(self.filter_kernel, ' d k -> d m k', m=1).float()
            x = F.conv1d(x, kernel, bias=None, padding='same', groups=x.shape[1])

        if not self.transposed: x = rearrange(x, 'b d l -> b l d')

        return x

    @property
    def d_output(self):
        return self.d_input

    def step(self, x, state, **kwargs):
        if self.pool > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state


""" Pooling functions with trainable parameters """ # TODO make d_output expand instead
class DownPool(SequenceModule):
    def __init__(self, d_input, d_output, pool, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()
        self._d_output = d_output
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * pool,
            d_output,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.pool)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs): # TODO needs fix in transpose ca, **kwargsse
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []

    @property
    def d_output(self): return self._d_output


class UpPool(SequenceModule): # TODO subclass SequenceModule
    def __init__(self, d_input, d_output, pool, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()
        self.d_input = d_input
        self._d_output = d_output
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            d_output * pool,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)
        if self.transposed:
            x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
            x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)
        else:
            x = F.pad(x[..., :-1, :], (0, 0, 1, 0)) # Shift to ensure causality
            x = rearrange(x, '... l (h s) -> ... (l s) h', s=self.pool)
        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state

    @property
    def d_output(self): return self._d_output

registry = {
    'sample': DownSample,
    'pool': DownPool,
}
