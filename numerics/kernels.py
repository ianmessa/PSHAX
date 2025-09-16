import jax
from jax import numpy as jnp
from flax import nnx

from collections.abc import Callable

from processing import uv_bump

AL_c = jnp.array([-0.1343, 0.0668, -0.4288, -0.0636, 0.0082])

def AL_krnl(x1x2):
    """Abrahamson/Lacour 21 kernel. Takes input of shape (2, 2) ((M1, R1), (M2, R2))."""
    assert x1x2.shape == (2, 2), "Bad shape..."
    dMw, dR = jnp.diff(x1x2, axis = 0)
    dlnR = jnp.log(jnp.abs(dR))
    y = AL_c[0] * dMw + AL_c[1] * dlnR + \
        jnp.exp(AL_c[2 * dMw ** 2 + AL_c[3] * dlnR + AL_c[4] * dMw * dlnR])
    return y

class nrl_krnl(nnx.Module):
    """A neural kernel with optional stationarity. Input is of shape (2, in_dim), where
    the first axis is (x1, x2). If stationary, runs L2 norm of x1 & x2 through feedforwards and
    takes absolute value. Otherwise, runs x1 & x2 through the same forward and takes product of absolute values afterward.
    At the end, we multiply the output by a tunable bump function along the L2 norm to make the kernel compactly supported.
    For high c-values, the bump function practically becomes a step function, but it's still smooth."""
    def __init__(self, in_dim:int, 
                       hidden_dim:int, 
                       hidden_num:int,
                       stationary:bool = True,
                       b:int = 1, c:int = 1,
                       activation:Callable = nnx.sigmoid, 
                       rngs:nnx.Rngs = nnx.Rngs(0, )):
        # Stationarity flag
        self.stationary = stationary
        # Activation fn
        self.activation = activation
        # Bump fn parameters
        self.b, self.c = nnx.Param(b), nnx.Param(c)
        # Feedforwards
        if self.stationary:
            in_dim = 1
        self.in_ffn = nnx.Linear(in_dim, hidden_dim, rngs = rngs)
        self.h_ffns = [nnx.Linear(hidden_dim, hidden_dim, rngs = rngs) for i in range(hidden_num)]
        self.out_ffn = nnx.Linear(hidden_dim, 1, rngs = rngs)

    def __call__(self, x1x2):
        """Pass x1/x2 of shape (2, in_dim) through kernel."""
        # Define x with bool. We can still jit this because the "if" is dependent on 
        #   a class variable.
        x_norm = jnp.linalg.norm(jnp.diff(x1x2, axis = 0), axis = 1)
        if self.stationary:
            x_in = x_norm
        else:
            x_in = x1x2
        # Run both through feedforwards
        x_in = self.activation(self.in_ffn(x_in))
        for h_ffn in self.h_ffns:
            x_in = self.activation(h_ffn(x_in))
        # Take absolute value to smooth
        x_in = jnp.abs(self.out_ffn(x_in))
        # If nonstationary, take product of each output
        if self.stationary:
            pass
        else:
            x_in = jnp.prod(x_in, axis = 0)
        # Apply bump fn to norm of x
        x_bump = uv_bump(x_norm, 1, self.b, self.c)
        y = x_in * x_bump
        return y