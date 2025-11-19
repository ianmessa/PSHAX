import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu

### ORTHOGONAL POLYNOMIALS ###
# Chebyshev (first kind)
def state_T(x:jax.Array) -> jax.Array:
    """First two terms of three-term recurrence for Chebyshev 
    polynomials of the first kind."""
    return x, jnp.ones_like(x), x

def ttr_T(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the first kind."""
    x, T_im1, T_i = state
    T_ip1 = 2 * x * T_i - T_im1
    return (x, T_i, T_ip1), T_ip1

# Chebyshev (second kind)
def state_U(x:jax.Array) -> jax.Array:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    return x, jnp.ones_like(x), 2 * x

def ttr_U(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    x, U_im1, U_i = state
    U_ip1 = 2 * x * U_i - U_im1
    return (x, U_i, U_ip1), U_ip1

# Legendre 
def state_P(x:jax.Array) -> jax.Array:
    """First two terms of the three-term recurrence for Legendre
    polynomials. """
    return x, jnp.ones_like(x), x

def ttr_P(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Legendre
    polynomials."""
    x, P_im1, P_i = state
    P_ip1 = ((2 * i + 1) * x * P_i - i * P_im1) / (i + 1)
    return (x, P_i, P_ip1), P_ip1

# Hermite
def state_H(x:jax.Array, alpha:float = 1) -> jax.Array:
    """First two terms of the three-term recurrence for generalized Hermite
    polynomials. """
    return x, jnp.ones_like(x), x / jnp.sqrt(alpha), alpha

def ttr_H(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for generalized Hermite
    polynomials."""
    x, H_im1, H_i, alpha = state
    H_ip1 = x * H_i - i * alpha * H_im1
    return (x, H_i, H_ip1), H_ip1

# Dictionary for all bases
psi_dict = {'T': {'state': state_T,
                  'ttr': ttr_T},
            'U': {'state': state_U,
                  'ttr': ttr_U},
            'P': {'state': state_P,
                  'ttr': ttr_P},
            'H': {'state': state_H,
                  'ttr': ttr_H}}

# Univariate orthogonal polynomial collection
# TODO: jit with staticargs basis, k_exc
def uv_psi(x:jax.Array, basis:str, k_exc:int, a:float = -1., b:float = 1.) -> jax.Array:
    """
    Generates a collection of univariate orthogonal polynomials evaluated at points x.

    Parameters
    ----------
    x : jax.Array
        Input points at which to evaluate the polynomials.
    basis : str
        Polynomial basis type. Must be one of 'T' (Chebyshev 1st kind), 
        'U' (Chebyshev 2nd kind), 'P' (Legendre), or 'H' (Hermite).
    k_exc : int
        Number of polynomial orders to generate (exclusive upper bound).
        The highest order generated will be k_exc - 1.
    a : float
        Lower bound on domain, default -1.
    b: float
        Upper bound on domain, default 1.

    Returns
    -------
    jax.Array
        Array of shape (k_exc,) containing the evaluated polynomials at x,
        from order 0 up to k_exc - 1.
    """
    # Our highest order will be k_exc - 1
    k = k_exc - 1

    # Scale x
    x_scaled = 2 * (x - a) / (b - a) - 1

    # Polynomial generation and return
    init_state = psi_dict[basis]['state'](x_scaled)
    ttr = psi_dict[basis]['ttr']
    k_range = jnp.arange(1, k)
    _, y = lax.scan(ttr, init_state, k_range)
    y = jnp.concatenate([jnp.array(init_state[1:]), y])

    return y

def mv_psi(X:jax.Array, basis:str, k_total:int, a:float | jax.Array = jnp.nan, b:float | jax.Array = jnp.nan) -> jax.Array:
    """
    Generates a collection of multivariate orthogonal polynomials evaluated at points X.

    Parameters
    ----------
    X : jax.Array
        Input points at which to evaluate the polynomials, shape (n, dim).
    basis : str
        Polynomial basis type. Must be one of 'T' (Chebyshev 1st kind), 
        'U' (Chebyshev 2nd kind), 'P' (Legendre), or 'H' (Hermite).
    k_total : int
        Maximum total polynomial order (sum of orders across dimensions).
    a : float | jax.Array
        Lower bound on domain, default NaN. If NaN, lower bound inferred from data.
        If float, broadcast to dim. If jax.Array, must be of shape (dim,).
    a : float | jax.Array
        Upper bound on domain, default NaN. If NaN, upper bound inferred from data.
        If float, broadcast to dim. If jax.Array, must be of shape (dim,).

    Returns
    -------
    jax.Array
        Array of shape (n, num_terms) containing the evaluated multivariate polynomials at X,
        where num_terms is the number of multi-indices with sum <= k_total.
    """

    n, dim = X.shape
    # Domain inference
    a_inf, b_inf = X.min(axis = 0), X.max(axis = 0)
    a_def, b_def = jnp.broadcast_to(a, dim), jnp.broadcast_to(b, dim)
    cond_a, cond_b= jnp.any(jnp.isnan(a)), jnp.any(jnp.isnan(b))
    a, b = lax.select(cond_a, a_inf, a_def), lax.select(cond_b, b_inf, b_def)
    X_scaled = 2 * (X - a) / (b - a) - 1

    # Generate univariate polynomials for each dimension
    Y = jax.vmap(jtu.Partial(uv_psi, basis = basis, k_exc = k_total + 1), in_axes = 1)(X_scaled)
    
    # Multiplication indices
    j = jnp.stack(jnp.meshgrid(*[jnp.arange(k_total + 1)] * dim), axis = -1).reshape((k_total + 1) ** dim, dim)
    j = j[j.sum(axis = 1) <= k_total]
    i = jnp.stack([jnp.arange(dim)] * j.shape[0])
    
    # Take product and return
    Y = jnp.prod(Y[i, j], axis = 1).T

    return Y