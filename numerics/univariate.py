import jax
from jax import lax
from jax import numpy as jnp
from jax.tree_util import Partial
from jax.typing import ArrayLike
from jax.experimental.jet import jet
from jax.experimental.checkify import check
jax.config.update('jax_enable_x64', True)

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

# Univariate normal distribution
def uv_normal(x:jax.Array, mu:float = 0, std:float = 1) -> jax.Array:
    """ A Gaussian PDF evaluated at x."""
    num = jnp.exp(-(x - mu) ** 2 / (2 * std ** 2))
    denom = (std * jnp.sqrt(2 * jnp.pi))
    return num / denom

def uv_H(x:jax.Array, k:int, mu:float = 0, std:float = 1, eps:float = 1e-100) -> jax.Array:
    """With x of shape n, returns (k, n) array of Hermite polynomials
    from orders 1 to k."""
    # Flatten x
    x_flat = x.squeeze()
    # Get primals (derivatives of f(x) = x)
    x_ser = (1.,) + (0., ) * (k - 1)

    # Define partial so we can take derivatives without passing
    #   mu, std
    normal_H = Partial(uv_normal, mu = mu, std = std)
    # Get nth derivatives using jax jet
    y, dny_dxn = jet(normal_H, (x_flat,), (x_ser,))
    dny_dxn = jnp.array(dny_dxn)
    # Take signs for each one
    signs = (-1) ** jnp.arange(1, k + 1)[:, None]
    
    # Update y to have all nonzero entries
    y = y.at[y == 0].set(eps)
    # Slam together as big (n, k) array
    return (dny_dxn * signs / (y))