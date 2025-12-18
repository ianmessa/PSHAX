import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu

### ORTHOGONAL POLYNOMIALS ###
# Nulls added for compatibility with Hermite scaling
# Chebyshev (first kind)
def state_T(x:jax.Array, null:float = 0) -> jax.Array:
    """First two terms of three-term recurrence for Chebyshev 
    polynomials of the first kind."""
    return x, jnp.ones_like(x), x, 0.

def ttr_T(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the first kind."""
    x, T_im1, T_i, _ = state
    T_ip1 = 2 * x * T_i - T_im1
    return (x, T_i, T_ip1, 0.), T_ip1

def state_T_mat(A:jax.Array, Omega:jax.Array) -> jax.Array:
    """First two terms of three-term recurrence for Chebyshev matrix
    polynomials of the first kind."""
    return A, Omega, A @ Omega, 0.

def ttr_T_mat(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev matrix
    polynomials of the first kind."""
    A, T_im1, T_i = state
    T_ip1 = 2 * A @ T_i - T_im1
    return (A, T_i, T_ip1), T_ip1

# Chebyshev (second kind)
def state_U(x:jax.Array, null:float = 0.) -> jax.Array:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    return x, jnp.ones_like(x), 2 * x, 0.

def ttr_U(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    x, U_im1, U_i, _ = state
    U_ip1 = 2 * x * U_i - U_im1
    return (x, U_i, U_ip1, 0.), U_ip1

def state_U_mat(A:jax.Array, Omega:jax.Array) -> jax.Array:
    """lax.scan-formatted three-term recurrence for Chebyshev matrix
    polynomials of the second kind."""
    return A, Omega, 2 @ A @ Omega

def ttr_U_mat(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    A, U_im1, U_i = state
    U_ip1 = 2 * A @ U_i - U_im1
    return (A, U_i, U_ip1), U_ip1

# Legendre 
def state_P(x:jax.Array, null:float = 0.) -> jax.Array:
    """First two terms of the three-term recurrence for Legendre
    polynomials. """
    return x, jnp.ones_like(x), x, 0.

def ttr_P(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Legendre
    polynomials."""
    x, P_im1, P_i, _ = state
    P_ip1 = ((2 * i + 1) * x * P_i - i * P_im1) / (i + 1)
    return (x, P_i, P_ip1, 0.), P_ip1

def state_P_mat(A:jax.Array, Omega:jax.Array) -> jax.Array:
    """First two terms of the three-term recurrence for Legendre matrix
    polynomials. """
    return A, Omega, A @ Omega

def ttr_P_mat(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Legendre
    polynomials."""
    A, P_im1, P_i, _ = state
    P_ip1 = ((2 * i + 1) @ A @ P_i - i @ P_im1) / (i + 1)
    return (A, P_i, P_ip1, 0.), P_ip1

# Hermite (no matrix TTR)
def state_H(x:jax.Array, alpha:float = 1) -> jax.Array:
    """First two terms of the three-term recurrence for generalized Hermite
    polynomials. """
    return x, jnp.ones_like(x), x / jnp.sqrt(alpha), alpha

def ttr_H(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for generalized Hermite
    polynomials."""
    x, H_im1, H_i, alpha = state
    H_ip1 = x * H_i - i * alpha * H_im1
    return (x, H_i, H_ip1, alpha), H_ip1

# Scaling functions
# To [-1, 1]
def scale_ones(x):
    return 2 * (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0)) - 1

def scale_spectral(A):
    centers = jnp.diag(A)
    radii = A.sum(axis = 1) - centers
    lmda_min = (radii - centers).min()
    lmda_max = (radii + centers).max()
    return (2 * A - (lmda_max + lmda_min)) / (lmda_max - lmda_min)

# Dictionary for all bases
psi_dict = {'T': {'state': state_T,
                  'ttr': ttr_T,
                  'state_mat': state_T_mat,
                  'ttr_mat': ttr_T_mat,
                  'scale': scale_ones},
            'U': {'state': state_U,
                  'ttr': ttr_U,
                  'state_mat': state_U_mat,
                  'ttr_mat': ttr_U_mat,
                  'scale': scale_ones},
            'P': {'state': state_P,
                  'ttr': ttr_P,
                  'state_mat': state_P_mat,
                  'ttr_mat': ttr_P_mat,
                  'scale': scale_ones},
            'H': {'state': state_H,
                  'ttr': ttr_H,
                  'scale': lambda x: x}}

# Univariate orthogonal polynomial collection
# TODO: jit with staticargs basis, k_exc
def uv_psi(x:jax.Array, basis:str, k_exc:int, alpha:float = 1.) -> jax.Array:
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
    alpha : float = 1
        For the Hermite basis, alpha controls std. This is different than merely stretching the 
        polynomials. For all other polynomials, alpha does nothing.

    Returns
    -------
    jax.Array
        Array of shape (n, k_exc) containing the evaluated polynomials at x,
        from order 0 up to k_exc - 1.
    """
    # Our highest order will be k_exc - 1
    k = k_exc - 1

    # Scale x
    scale_fn = psi_dict[basis]['scale']
    x_scaled = scale_fn(x)

    # Polynomial generation and return
    init_state = psi_dict[basis]['state'](x_scaled, alpha)
    ttr = psi_dict[basis]['ttr']
    k_range = jnp.arange(1, k)
    _, y = lax.scan(ttr, init_state, k_range)
    y = jnp.concatenate([jnp.array(init_state[1:-1]), y])

    return y.T

def mv_psi(X:jax.Array, basis:str, k_total:int, alpha:jax.Array) -> jax.Array:
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
    alpha : jax.Array
        For the Hermite basis, alpha controls std. This is different than merely stretching the 
        polynomials. Must of shape (dim,). 

    Returns
    -------
    jax.Array
        Array of shape (n, num_terms) containing the evaluated multivariate polynomials at X,
        where num_terms is the number of multi-indices with sum <= k_total.
    """

    n, dim = X.shape
    # Domain inference
    scale_fn = psi_dict[basis]['scale']
    X_scaled = scale_fn(X)

    alpha = jnp.atleast_1d(alpha)
    alpha = jnp.broadcast_to(alpha, (dim,))

    # Generate univariate polynomials for each dimension of shape (dim, n, k_total + 1)
    # Lambda fn for easy vmap
    uv_psi_partial = jtu.Partial(uv_psi, basis = basis, k_exc = k_total + 1)
    Y = jax.vmap(lambda X_scaled, alpha: uv_psi_partial(X_scaled, alpha = alpha), in_axes = (1, 0), out_axes = 1)(X_scaled, alpha)
    
    # Multiplication indices
    j = jnp.stack(jnp.meshgrid(*[jnp.arange(k_total + 1)] * dim), axis = -1).reshape((k_total + 1) ** dim, dim)
    j = j[j.sum(axis = 1) <= k_total]
    i = jnp.stack([jnp.arange(dim)] * j.shape[0])
    
    # Take product and return
    Y = jnp.prod(Y[:, i, j], axis = -1)

    return Y

def mat_psi(A:jax.Array, Omega:jax.Array, basis:str, k_exc:int) -> jax.Array:
    """
    Generates a collection of univariate orthogonal polynomials evaluated at points x.

    Parameters
    ----------
    A : jax.Array
        Input matrix (upon?... at?...) which to evaluate the polynomials
        of shape (n, n).
    Omega: jax.Array
        Initial matrix (instead of a constant) used to construct polynomials
        (first term is Omega instead of a constant matrix) of shape (n, d).
    basis : str
        Polynomial basis type. Must be one of 'T' (Chebyshev 1st kind), 
        'U' (Chebyshev 2nd kind), or 'P' (Legendre).
    k_exc : int
        Number of polynomial orders to generate (exclusive upper bound).
        The highest order generated will be k_exc - 1.

    Returns
    -------
    jax.Array
        Array of shape (n, n, k_exc) containing the evaluated polynomials at x,
        from order 0 up to k_exc - 1.
    """
    # Our highest order will be k_exc - 1
    k = k_exc - 1

    # Scale x
    A_scaled = scale_spectral(A)

    # Polynomial generation and return
    init_state = psi_dict[basis]['state_mat'](A_scaled, Omega)
    ttr = psi_dict[basis]['ttr_mat']
    k_range = jnp.arange(1, k)
    _, y = lax.scan(ttr, init_state, k_range)
    y = jnp.concatenate([jnp.array(init_state[1:-1]), y])

    return y.T