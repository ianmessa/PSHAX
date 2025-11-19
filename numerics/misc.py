import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental.jet import jet
from jax.scipy.stats import norm as normal

### UNIVARIATE FUNCTIONS ###
# Univariate bump function.
def uv_bump(x:jax.Array, a:float = 1, b:float= 1, c:float = 10) ->jax.Array:
    """ Bump function normalized to 1-max. 
    x is input. 
    a, b, c, d for amplitude, width, shape."""
    b, c = [jnp.clip(p, min = 1E-2) for p in [b, c]]
    f = a * jnp.exp(c ** (-2) + (b ** 2) / (c ** 2 * (((x) ** 2) - b ** 2)))
    cond = jnp.any(jnp.stack([x < -b, 
                              x > b, 
                              jnp.isinf(jnp.abs(f))]), 
                        axis = 0)
    f = jnp.where(cond, 0, f)
    return f