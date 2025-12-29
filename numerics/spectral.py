import jax
from jax import lax
from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax import random as jrnd
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
    return (x, H_i, H_ip1, alpha), H_ip1

# Scaling functions to [-1, 1]
def scale_ones(x):
    return 2 * (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0)) - 1

# Dictionary for all bases
psi_dict = {'T': {'state': state_T,
                  'ttr': ttr_T,
                  'scale': scale_ones},
            'U': {'state': state_U,
                  'ttr': ttr_U,
                  'scale': scale_ones},
            'P': {'state': state_P,
                  'ttr': ttr_P,
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

# Multivariate Orthopoly
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
    j = jnp.stack(jnp.meshgrid(*[jnp.arange(k_total)] * dim), axis = -1).reshape((k_total) ** dim, dim)
    j = j[j.sum(axis = 1) <= k_total]
    i = jnp.stack([jnp.arange(dim)] * j.shape[0])
    
    # Take product and return
    Y = jnp.prod(Y[:, i, j], axis = -1)

    return Y

# Matrix Orthopoly
# Only using Chebyshev First Kind
def state_mat(A:jax.Array, B:jax.Array) -> jax.Array:
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

def ttr_mat(state:tuple, i:int) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
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

# Scale spectrum to [-1, 1]
def scale_spectrum(A:jax.Array, 
                    max_iter:int = 50,
                    tol:float = 1e-4, 
                    key:jax.Array = jrnd.key(0)):
    """Scale the spectrum of a PSD symmetric matrix to ~[-1, 1].

    Parameters
    ----------
    A : jax.Array
        The input positive semi-definite (PSD) symmetric matrix to be scaled.
    max_iter : int, optional
        Maximum number of iterations for the power iteration method (default is 50).
    tol : float, optional
        Tolerance for convergence in the power iteration (default is 1e-4).
    key : jax.Array, optional
        Random key for initial vector (default is jrnd.key(0)).

    Returns
    -------
    tuple[jax.Array, float]
        A tuple containing the scaled matrix (with spectrum approximately in [-1, 1]) 
        and the estimated upper bound of the maximum eigenvalue.
    """
    n = A.shape[0]

    # Power iteration for maximum eigenvalue
    def pow_cond(state):
        b_i, b_im1, i = state
        lmda_ceil = (b_im1 @ b_i)
        conv_cond = jnpla.norm(b_i - lmda_ceil * b_im1) > tol
        max_cond = i < max_iter
        return jnp.logical_and(conv_cond, max_cond)
    
    def pow_iter(state):
        b_i, b_im1, i = state
        b_ip1 = A @ b_i
        b_ip1 = b_ip1 / jnpla.norm(b_ip1)
        return b_ip1, b_i, i + 1
    
    # Initial state
    b0 = jrnd.ball(key, d = n, p = 1)
    b1 = A @ b0
    b1 = b1 / jnpla.norm(b1)
    # Scale both vectors
    # Keep the last two
    bf, b_fm1, _ = lax.while_loop(pow_cond, pow_iter, (b1, b0, 0))
    # Calculate the maximum eigenvalue...
    lmda_ceil = bf.T @ A @ bf
    rem = jnpla.norm(A @ bf - lmda_ceil * bf)
    # And add remainder to get a definite upper bound
    lmda_ceil = lmda_ceil + rem

    # Return the scaled matrix
    return (2 * A / lmda_ceil) - jnp.eye(n), lmda_ceil

# Block Krylov for Chebyshev Matrix Polynomials
def mat_psi(A:jax.Array, Omega:jax.Array, k_exc:int):
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
    init_state = state_mat(A, Omega)
    # Right now, y is poly ranging from [2, q-1]
    _, y = lax.scan(ttr_mat, init_state, jnp.arange(k_exc - 2))
    # Add the initial state onto the front (zeroth and first terms)
    y = jnp.concatenate([init_state[-1][None], init_state[-2][None], y], axis = 0)

    # Transpose to shape (n, m, q)
    return y.transpose(1, 0, 2)

def rBK(A:jax.Array, n_eigs:int, 
    k_exc:int,
    max_iter:int = 50,
    tol:float = 1e-8,
    key:jax.Array = jrnd.key(0)):
    """
    Randomized block-Krylov method for computing the top n_eigs eigenpairs of a symmetric positive semidefinite matrix 
    using Chebyshev polynomial. We scale the spectrum of the input matrix; randomly sample its columns; produce a block Krylov basis
    for the random sample; orthogonalize the basis; apply it to the input matrix; and extract its eigenvalue and eigenvetors.

    Parameters
    ----------
    A : jax.Array
        Symmetric positive semidefinite matrix of shape (n, n).
    n_eigs : int
        Number of top eigenvalues and eigenvectors to compute.
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
    A0 = A

    # Scale spectrum to (-1, 1)
    A, lmda_ceil = scale_spectrum(A, 
                  max_iter, 
                  tol, 
                  keys[0])

    # Random matrix
    Omega = jrnd.normal(keys[1], (n, n_eigs))

    # Build orthogonalized Krylov basis
    Q = mat_psi(A, Omega, k_exc).reshape(n, -1)
    Q, _ = jnpla.qr(Q)

    # Drop A to orthogonal basis
    # (A is symmetric so this will be, too)
    M = Q.T @ A @ Q

    # Extract eigs
    lmda, U = jnpla.eigh(M)
    # Rescale after scaling for Chebyshev filter
    lmda = (lmda + 1) * (lmda_ceil / 2)
    # Lift eigenvectors back to original subspace
    V = Q @ U

    # Truncate oversampled results
    trunc = n_eigs * (q - 1)
    lmda, V = lmda[trunc:], V[:, trunc:]
    return lmda, V