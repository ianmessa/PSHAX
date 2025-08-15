import jax
from jax import numpy as jnp
from jax.experimental.jet import jet
from jax.tree_util import Partial

from scipy.special import roots_legendre
from collections.abc import Callable

# Univariate normal distribution
def uv_normal(mu:float, std:float, x:jax.Array) -> jax.Array:
    """ A univariate normal distribution evaluated at x."""
    num = jnp.exp(-(x - mu) ** 2 / (2 * std ** 2))
    denom = (std * jnp.sqrt(2 * jnp.pi))
    return num / denom

# Univariate bump function.
def uv_bump(x, a:float = 1, b:float= 1, c:float = 1, d:float = 0):
    """ Bump function normalized to 1-max. 
    x is input. 
    a, b, c, d for amplitude, width, shape, and displacement, respectively."""
    b, c = [jnp.clip(p, min = 1E-2) for p in [b, c]]
    f = a * jnp.exp(c ** (-2) + (b ** 2) / (c ** 2 * (((x - d) ** 2) - b ** 2)))
    cond = jnp.any(jnp.stack([x < (d - b), 
                              x > (d + b), 
                              jnp.isinf(jnp.abs(f))]), 
                        axis = 0)
    f = jnp.where(cond, 0, f)
    return f

# Univariate hermite
def uv_H(k:int, mu:float, std:float, x:jax.Array) -> jax.Array:
    """ kth-order probabilist's Hermite polynomial fit to N(mu, std) and evaluated at x."""
    # Flatten x
    x_flat = x.squeeze()
    # Get primals (derivatives of f(x) = x)
    x_ser = (1.,) + (0., ) * (k - 1)

    # Define partial so we can take derivatives without passing
    #   mu, std
    normal_H = Partial(uv_normal, mu, std)
    # Get nth derivatives using jax jet
    y, dy = jet(normal_H, (x_flat,), (x_ser,))
    dy = jnp.array(dy)
    # Add primal for H0
    d_normal_H = jnp.insert(dy, 0, y, axis = 0)
    # Take signs for each one
    signs = (-1) ** jnp.arange(0, k + 1)[:, None]
    
    # Slam together as big l*(n + 1) array (l is number of entries, 
    #   n is hermite pol. degree (n + 1 b/c of H0))
    return (d_normal_H * signs / normal_H(x_flat)).T

# multivariate hermite family
def mv_H(k:int, mu:jax.Array, std:jax.Array, x:jax.Array) -> jax.Array:
    """mu and std are m-dimensional arrays. x is (n, m). n specifies the maximum
     order of the Legendre polynomial. Returns all multivariate Hermite polynomials defined
     over the specified multivariate Gaussian--that is, all possible products of order <= k."""
    m = mu.shape[0]
    assert std.shape[0] == m and x.shape[-1] == m, "Dimensions incompatible..."
    assert len(x.shape) == 2, "X must be an  array."
    assert len(mu.shape) == 1 and len(std.shape) == 1, "Mu and StD must be 1-dimensional."
    # First dimension (variate) index
    i = jnp.stack([jnp.arange(m)] * ((k + 1) ** m), axis = -1)
    # Second dimension (order) index
    #   This is a cartesian product of all H_Dn.
    #   In this case underscore = subscript.
    j = jnp.stack(jnp.meshgrid(*[jnp.arange(k + 1)] * m), axis = 0).reshape(m, (k + 1) ** m)
    # Produce all hermite polynomials <= n for all variables (dimensions of x, m)
    y_tens = jax.vmap(Partial(H, k))(mu, std, x.T).transpose(0, 2, 1)
    # Take groupwise products
    y = jnp.prod(y_tens[i, j], axis = 0)
    return y

# Legendre roots + weights (REPLACE...).
def L_rw(k:int):
    """ All roots + weights for kth-order Legendre polynomial."""
    x, w = [jnp.array(v) for v in roots_legendre(k)]
    return x, w

# Fixed-order m-dimensional Gauss-Legendre Quadrature
def fixed_GLq(f:Callable, D:jax.Array, k:int):
    """Fixex-order m-dimensional Gauss-Legendre Quadrature.
    If D is an mx2 array, f takes an (m,)-shaped vector of arguments.
    We evaluate it at the cartesian product of the roots of m scaled
    nth-order Legendre polynomials. """
    assert len(D.shape) == 2 and D.shape[1] == 2, "D must be an mx2 array if f is a scalar function of m arguments."
    m = D.shape[0]
    # Get roots and weights
    x, w = L_rw(k)
    # Produce x/w grids
    X = jnp.stack(jnp.meshgrid(*[x] * m), axis = -1).reshape(k ** m, m)
    W = jnp.prod(jnp.stack(jnp.meshgrid(*[w] * m)), axis = 0).reshape(k ** m, 1)
    
    # Define scaling + shifting
    scale = ((D[:, 1] - D[:, 0]) / 2)[None]
    shift = ((D[:, 1] + D[:, 0]) / 2)[None]
    X = scale * X + shift
    
    # Final scaling and dot between y and W
    return jnp.prod(scale) * (f(X) @ W)

# Sobol sequence! 
def sobol(d, pn):
    """d-dimensional Sobol sequence of 2^pn points."""
    
