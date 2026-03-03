import jax
from jax import lax
from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax import random as jrnd
from jax import tree_util as jtu

### ORTHOGONAL POLYNOMIALS ###
# Nulls added for compatibility with Hermite scaling
# Monomial
def _state_M(x:jax.Array, null:float = 0) -> jax.Array:
    """Three-term recurrence(?... Just designed to fit the same framework) for monomial basis"""
    return x, jnp.ones_like(x), x, 0.

def _ttr_M(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for monomial basis."""
    x, M_im1, M_i, _ = state
    M_ip1 = M_i * x
    return (x, M_i, M_ip1, 0.), M_ip1

# Chebyshev (first kind)
def _state_T(x:jax.Array, null:float = 0) -> jax.Array:
    """First two terms of three-term recurrence for Chebyshev 
    polynomials of the first kind."""
    return x, jnp.ones_like(x), x, 0.

def _ttr_T(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the first kind."""
    x, T_im1, T_i, _ = state
    T_ip1 = 2 * x * T_i - T_im1
    return (x, T_i, T_ip1, 0.), T_ip1
    

# Chebyshev (second kind)
def _state_U(x:jax.Array, null:float = 0.) -> jax.Array:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    return x, jnp.ones_like(x), 2 * x, 0.

def _ttr_U(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev
    polynomials of the second kind."""
    x, U_im1, U_i, _ = state
    U_ip1 = 2 * x * U_i - U_im1
    return (x, U_i, U_ip1, 0.), U_ip1

# Legendre 
def _state_P(x:jax.Array, null:float = 0.) -> jax.Array:
    """First two terms of the three-term recurrence for Legendre
    polynomials. """
    return x, jnp.ones_like(x), x, 0.

def _ttr_P(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Legendre
    polynomials."""
    x, P_im1, P_i, _ = state
    P_ip1 = ((2 * i + 1) * x * P_i - i * P_im1) / (i + 1)
    return (x, P_i, P_ip1, 0.), P_ip1

# Hermite
def _state_H(x:jax.Array, alpha:float = 1) -> jax.Array:
    """First two terms of the three-term recurrence for generalized Hermite
    polynomials. """
    return x, jnp.ones_like(x), x / jnp.sqrt(alpha), alpha

def _ttr_H(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for generalized Hermite
    polynomials."""
    x, H_im1, H_i, alpha = state
    H_ip1 = x * H_i - i * alpha * H_im1
    return (x, H_i, H_ip1, alpha), H_ip1

# Scaling functions to [-1, 1]
def scale_ones(x):
    return 2 * (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0)) - 1

# Dictionary for all bases
psi_dict = {'M': {'state': _state_M,
                  'ttr': _ttr_M,
                  'scale': lambda x: x},
            'T': {'state': _state_T,
                  'ttr': _ttr_T,
                  'scale': scale_ones},
            'U': {'state': _state_U,
                  'ttr': _ttr_U,
                  'scale': scale_ones},
            'P': {'state': _state_P,
                  'ttr': _ttr_P,
                  'scale': scale_ones},
            'H': {'state': _state_H,
                  'ttr': _ttr_H,
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
        Polynomial basis type. Must be one of 'M' (Monomial), 'T' (Chebyshev 1st kind), 
        'U' (Chebyshev 2nd kind), 'P' (Legendre), or 'H' (Hermite).
    k_exc : int
        Number of polynomial orders to generate (exclusive upper bound).
        The highest order generated will be k - 1.
    alpha : float = 1
        For the Hermite basis, alpha controls std. This is different than merely stretching the 
        polynomials. For all other polynomials, alpha does nothing.

    Returns
    -------
    jax.Array
        Array of shape (n, k_exc) containing the evaluated polynomials at x,
        from order 0 up to k_exc - 1.
    """
    # Scale x
    scale_fn = psi_dict[basis]['scale']
    x_scaled = scale_fn(x)

    # Polynomial generation and return
    init_state = psi_dict[basis]['state'](x_scaled, alpha)
    ttr = psi_dict[basis]['ttr']
    k_range = jnp.arange(1, k_exc - 1)
    _, y = lax.scan(ttr, init_state, k_range)
    y = jnp.concatenate([jnp.array(init_state[1:-1]), y])

    return y.T

def _summask(j, k_max, q):
    return j.sum(axis = -1) <= k_max
def _prodmask(j, k_max, q):
    return jnp.prod(j, axis = -1) <= k_max
def _hbmask(j, k_max, q):
    return jnp.power(j, q).sum(axis = -1) ** (1 / q) <= k_max

def _combomask(j):
    j_unique = jnp.unique(jnp.sort(j, axis = -1), axis = 0, size = j.shape[0], fill_value = j.max() + 1)
    return jnp.all(jnp.isin(j_unique, j), axis = -1)
def _permmask(j):
    return jnp.ones(j.shape[0], dtype = bool)

_strategies = [_summask, _prodmask, _hbmask]
_strategy_idcs = {'sum':0, 'prod':1, 'hyperbolic':2}

def _trunc_mask(j:jax.Array, 
                k_max:int, strategy:str = 'sum', 
                order_matters:bool = True,
                q:float = 0.625):
    strategy_idx = _strategy_idcs[strategy]
    strat_mask = lax.switch(strategy_idx, _strategies, j, k_max, q)
    order_mask = lax.cond(order_matters, _permmask, _combomask, j)
    mask = jnp.logical_and(strat_mask, order_mask)
    return mask

# Multivariate Orthopoly
def mv_psi(x:jax.Array, basis:str, 
           k_max:int, strategy:str = 'sum', q:float = 0.75,
           order_matters:bool = True,
           alpha:float | jax.Array = 0.) -> jax.Array:
    """
    Generates a collection of multivariate orthogonal polynomials evaluated at points X.

    Parameters
    ----------
    x : jax.Array
        Input points at which to evaluate the polynomials, shape (n, dim).
    basis : str
        Polynomial basis type. Must be one of 'M' (Monomial), 'T' (Chebyshev 1st kind), 
        'U' (Chebyshev 2nd kind), 'P' (Legendre), or 'H' (Hermite).
    k_max : int
        Maximum polynomial order depending on truncation strategy.
    strategy : str
        Max-sum/max-product/hyperbolic truncation; the last one uses q.
    q : float
        Power for hyperbolic truncation
    alpha : FLOAT | jax.Array
        For the Hermite basis, alpha controls std. This is different than merely stretching the 
        polynomials. Must broadcastable to shape (dim,). 

    Returns
    -------
    jax.Array
        Array of shape (n, k_exc ** dim).
    """

    n, dim = x.shape
    alpha = jnp.broadcast_to(alpha, (dim,))
    # Domain inference
    scale_fn = psi_dict[basis]['scale']
    X_scaled = scale_fn(x)

    alpha = jnp.atleast_1d(alpha)
    alpha = jnp.broadcast_to(alpha, (dim,))

    # Generate univariate polynomials for each dimension of shape (dim, n, k_exc)
    # Lambda fn for easy vmap
    uv_psi_partial = jtu.Partial(uv_psi, basis = basis, k_exc = k_max + 1)
    Y = jax.vmap(lambda X_scaled, alpha: uv_psi_partial(X_scaled, alpha = alpha), in_axes = (1, 0), out_axes = 1)(X_scaled, alpha)
    
    j = jnp.indices((k_max,) * dim).reshape(dim, -1).T
    i = jnp.broadcast_to(jnp.arange(dim), j.shape)
    mask = _trunc_mask(j, k_max, strategy, order_matters, q)
    i, j = i[mask], j[mask]

    Y = jnp.prod(Y[:, i, j], axis = -1)

    return Y

# Matrix Orthopoly
# Only using Chebyshev First Kind
def _state_mat(A:jax.Array, B:jax.Array) -> jax.Array:
    """First two terms of three-term recurrence for Chebyshev matrix
    polynomials of the first kind evaluated using A with B as the first term.

    Parameters
    ----------
    A : jax.Array
        The matrix used in the recurrence relation.
    B : jax.Array
        The initial matrix (first term) in the recurrence.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        A tuple containing A, A @ B, and B.
    """

    return A, A @ B, B

def _ttr_mat(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """lax.scan-formatted three-term recurrence for Chebyshev matrix
    polynomials of the first kind.

    Parameters
    ----------
    state : tuple[jax.Array, jax.Array, jax.Array]
        A tuple containing the matrix A, the current polynomial term T_i, 
        and the previous polynomial term T_im1.
    i : int
        The current iteration index (not used in the computation).

    Returns
    -------
    tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]
        A tuple containing the updated state (A, T_ip1, T_i) and the next 
        polynomial term T_ip1.
    """
    A, T_i, T_im1 = state
    n = A.shape[0]
    T_ip1 = 2 * A @ T_i - T_im1
    return (A, T_ip1, T_i), T_ip1

def _scale_spectrum(A: jax.Array, 
                            max_iter: int = 50,
                            tol: float = 1e-4, 
                            key: jax.Array = jrnd.key(0)):
    n = A.shape[0]

    def pow_cond(state):
        b_i, b_im1, i = state
        # Power iteration
        conv_cond = jnpla.norm(b_i - (b_i @ b_im1) * b_im1) > tol
        return jnp.logical_and(conv_cond, i < max_iter)
    
    def pow_iter(state):
        b_i, _, i = state
        b_next = A @ b_i
        return b_next / jnpla.norm(b_next), b_i, i + 1
    
    b0 = jrnd.normal(key, (n,))
    b0 = b0 / jnpla.norm(b0)
    bf, _, _ = lax.while_loop(pow_cond, pow_iter, (b0, b0, 0))

    # Rayleigh quotient
    rho_est = jnp.abs(bf.T @ A @ bf)
    # Add eps
    rho_ceil = rho_est * 1.01

    # Map [ -rho_ceil, rho_ceil ] to [ -1, 1 ]
    return A / rho_ceil, rho_ceil

# Block Krylov for Chebyshev Matrix Polynomials
def _mat_psi(A:jax.Array, Omega:jax.Array, k_exc:int):
    """
    Generates a collection of univariate orthogonal polynomials evaluated at points x.

    Parameters
    ----------
    A : jax.Array
        Input matrix (upon?... at?...) which to evaluate the polynomials
        of shape (n, n).
    Omega: jax.Array
        Initial matrix (instead of a constant) used to construct polynomials
        (first term is Omega instead of a constant matrix) of shape (n, m).
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
    # Polynomial generation and return
    init_state = _state_mat(A, Omega)
    # Right now, y is poly ranging from [2, q-1]
    _, y = lax.scan(_ttr_mat, init_state, jnp.arange(k_exc - 2))
    # Add the initial state onto the front (zeroth and first terms)
    y = jnp.concatenate([init_state[-1][None], init_state[-2][None], y], axis = 0)

    # Transpose to shape (n, m, q)
    return y.transpose(1, 0, 2)

def rBK(A:jax.Array, n_eigs:int, m_os:float,
    k_exc:int,
    max_iter:int = 50,
    tol:float = 1e-8,
    key:jax.Array = jrnd.key(0)):
    """
    Randomized Block-Krylov method for computing the top n_eigs eigenpairs of a symmetric positive semidefinite matrix 
    using Chebyshev polynomial. We scale the spectrum of the input matrix; randomly sample its columns; produce a block Krylov basis
    for the random sample; orthogonalize the basis; apply it to the input matrix; and extract its eigenvalue and eigenvetors.

    Parameters
    ----------
    A : jax.Array
        Symmetric positive semidefinite matrix of shape (n, n).
    n_eigs : int
        Number of top eigenvalues and eigenvectors to compute.
    m_os : int
        Oversampling factor.
    k : int
        Degree of the Chebyshev polynomial used in the Krylov subspace construction.
    max_iter : int, optional
        Maximum number of iterations for spectral scaling (default is 50).
    tol : float, optional
        Convergence tolerance (Av - λv) for spectral scaling (default is 1e-8).
    key : jax.Array, optional
        Random key for Gaussian matrix Ω.

    Returns
    -------
    tuple[jax.Array, jax.Array]
    A tuple containing:
    - lmda : jax.Array
        Array of shape (n_eigs,) containing the approximated top eigenvalues.
    - V : jax.Array
        Array of shape (n, n_eigs) containing the corresponding eigenvectors.

    Notes
    -----
    - The method oversamples by constructing a subspace of size n_eigs * q, then truncates to retain only the top n_eigs eigenpairs.
    - Eigenvalues are rescaled back to the original spectrum after filtering.
    - Assumes A is symmetric and PSD; no checks are performed internally.
    """
    keys = jrnd.split(key, 2)
    n = A.shape[0]
    n_os = int(n_eigs * (m_os))

    # Scale spectrum to (-1, 1)
    A, rho_ceil = _scale_spectrum(A, 
                  max_iter, 
                  tol, 
                  keys[0])

    # Random matrix
    Omega = jrnd.normal(keys[1], (n, n_os))

    # Build orthogonalized Krylov basis
    Q = _mat_psi(A, Omega, k_exc).reshape(n, -1)
    Q, _ = jnpla.qr(Q)

    # Drop A to orthogonal basis
    # (A is symmetric so this will be, too)
    M = Q.T @ A @ Q

    # Extract eigs
    lmda, U = jnpla.eig(M)
    # Rescale after scaling for Chebyshev filter
    lmda = lmda.real * rho_ceil
    # Lift eigenvectors back to original subspace
    V = Q @ U

    # Truncate oversampled results
    lmda, V = lmda[:n_eigs], V[:, :n_eigs]
    return lmda, V