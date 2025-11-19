import jax
from jax import numpy as jnp
from jax import random as jrnd
from jax import tree_util as jtu

from collections.abc import Callable

from .spectral import *
from .sobol import *

# Legendre roots
def rw_P(k:int, tol:float = 1e-14):
    """
    Generates k roots from Legendre polynomials using Newton's method. 

    Parameters
    ----------
    k : int
        Number of roots. 
    tol : float, optional
        Absoulte tolerance for Newton-Raphson

    Returns
    -------
    jax.Array
        Array of shape (k) containing roots of kth Legendre polynomial.
    """
    # uv_psi autoscales x without a domain input, so here's our domain
    D = jnp.array([-1., 1.])

    # Initial Cheb guess
    k_range = jnp.arange(k) + 1
    x0 = jnp.cos(jnp.pi * (4 * k_range - 1) / (4 * k + 2))[::-1]
    # Each time, k_exc will be k + 1 so we can grab the kth polynomial.
    P_km1, P_k = uv_psi(x0, basis = 'P', k_exc = k + 1)[-2:]
    init_state = (x0, P_km1, P_k)
    
    # Newton raphson
    def body_fn(state):
        x, P_km1, P_k = state
        # Analytical derivative
        dPdx_k = k * (P_km1 - x * P_k) / (1 - x ** 2)
        x = x - (P_k / dPdx_k)
        P = uv_psi(x, basis = 'P', k_exc = k + 1)
        P_km1, P_k= P[-2:]
        return (x, P_km1, P_k)
    
    def cond_fn(state):
        x, P_nm1, P_n = state
        return jnp.all(jnp.abs(P_n) > tol)
    
    # Run
    x, P_nm1, P_n, = lax.while_loop(cond_fn, body_fn, init_state)
    w = (2 * (1 - x ** 2)) / (k ** 2 * P_nm1 ** 2)

    return x, w

# Univariate Gauss-Legendre Quadrature 
# TODO: jit with staticargs n, tol
def uv_GL_quad(F:Callable, a:float, b:float, n:int, tol:float = 1e-14):
    """
    Univariate Gauss-Legendre Quadrature. 

    Parameters
    ----------
    F : Callable
        A scalar-input/scalar-output vectorizable function.
    a : float
        Lower integration bound.
    b: float
        Upper integration bound. 
    n: int
        Number of quadrature points.
    tol : float, optional
        Absoulte tolerance for Newton-Raphson

    Returns
    -------
    y: The integral of the function over [a, b]. 
    """
    # Obvious
    r, w = rw_P(n, tol = tol)

    r_scaled = r * (b - a) / 2 + (a + b) / 2
    w_scaled = w * (b - a) / 2

    return jax.vmap(F)(r_scaled) @ w_scaled

# Multivariate Gauss-Legendre Quadrature
# TODO: jit with staticargs dim, n, tol
def mv_GL_quad(F:Callable, dim:int, a:float|jax.Array, b:float|jax.Array, n:int, tol:float = 1e-14):
    """
    Multivariate Gauss-Legendre Quadrature. We use the same # of quadrature points
    in all dimensions.

    Parameters
    ----------
    F : Callable
        A function that takes array argument of shape (dim,) and returns a float.
    dim : int
        Number of dimensions for integration.
    a : float | jax.Array
        Lower integration bound. If float, broadcasted to shape (dim,), otherwise
        must be of shape (dim,).
    b : float | jax.Array
        Upper integration bound. If float, broadcasted to shape (dim,), otherwise
        must be of shape (dim,).
    n : int
        Number of quadrature points (must be divisible by dimension).
    tol : float, optional
        Absoulte tolerance for Newton-Raphson

    Returns
    -------
    y: The integral of the function over [a, b]. 
    """
    check(n % dim == 0, 'Number of points must be divisible by dimension...')
    # Take roots, weights (TODO: using floor because we'll probably get rid of that check)
    r, w = rw_P(n // dim, tol = tol)

    # Broadcast a and b
    a, b = jnp.broadcast_to(a, dim), jnp.broadcast_to(b, dim)

    # Tensorize
    R = jnp.stack(jnp.meshgrid(*[r] * dim), axis = -1).reshape((n // dim) ** dim, dim)
    W = jnp.stack(jnp.meshgrid(*[w] * dim), axis = -1).reshape((n // dim) ** dim, dim)

    # Scale tensors
    R_scaled = R * (b - a) / 2 + (a + b) / 2
    W_scaled = W * (b - a) / 2

    # Take tensor product for weights
    W_scaled = jnp.prod(W_scaled, axis = 1)

    # Return
    return jax.vmap(F)(R_scaled) @ W_scaled

# TODO: jit with staticargs dim, n, trials
def mv_RQMC_quad(F:callable, dim:int, a:float|jax.Array, b:float|jax.Array, n:int, trials:int, key:jax.Array):
    """
    Multivariate Randomized Quasi-Monte Carlo (Sobol) Quadrature. 

    Parameters
    ----------
    F : Callable
        A function that takes array argument of shape (dim,) and returns a float.
    dim : int
        Number of dimensions for integration.
    a : float | jax.Array
        Lower integration bound. If float, broadcasted to shape (dim,), otherwise
        must be of shape (dim,).
    b : float | jax.Array
        Upper integration bound. If float, broadcasted to shape (dim,), otherwise
        must be of shape (dim,).
    n : int
        Number of quadrature points (must be divisible by dimension).
    trials : int
        Number of randomized integrations to perform.
    key: jax.Array
        PRNG key.

    Returns
    -------
    Y: The integral of the function over [a, b] averaged across trials
    """
    # Generate sequence
    X = sobol(dim, n, a, b)
    # Apply Cranley-Patterson
    X_scrambled = jax.vmap(jtu.Partial(CP_rotation, X = X, a = a, b = b))(key = jrnd.split(key, trials))

    # Uniform weights
    W = jnp.prod(b - a) / n
    
    # Evaluate F at all points
    Y = jax.vmap(jax.vmap(F))(X_scrambled)

    # Return average
    return Y.sum() * W / trials

# Multivariate roots and weights
def mvrw_GL(dim:int, a:float|jax.Array, b:float|jax.Array, n:int):
    """
    Tensor-product GLQ rule.

    Parameters
    ----------
    dim : int
        Number of dimensions for the multivariate quadrature (must be >= 1).
    a, b : float
        Endpoints of the target integration interval [a, b] for each coordinate.
        Coordinates are affinely mapped from the reference interval [-1, 1].
    n : int
        Number of 1‑D Gauss–Legendre nodes/weights (as produced by rw_P).
        The total number of multivariate nodes will be n**dim.

    Returns
    -------
    rr : jax.numpy.ndarray, shape (n**dim, dim)
        Array of multivariate quadrature nodes. Each row is a 'dim'‑vector
        corresponding to a tensor‑product node mapped to the interval [a, b].
    ww : jax.numpy.ndarray, shape (n**dim,)
        Array of product weights for the tensor‑product rule. Weights are scaled
        by ((b - a) / 2)**dim and obtained by the elementwise product of the
        mapped 1‑D weights for each node.
    """
    
    # Get Legendre RW
    r, w = rw_P(n)
    # Cartesian product
    rr = jnp.stack(jnp.meshgrid(*[r] * dim), axis = -1).reshape(n ** dim, dim)
    ww = jnp.stack(jnp.meshgrid(*[w] * dim), axis = -1).reshape(n ** dim, dim)
    # Stretch and scale
    rr = rr * (b - a) / 2 + (a + b) / 2
    ww = ww * (b - a) / 2
    # Prod roots
    ww = jnp.prod(ww, axis = 1)
    return rr, ww

def mvrw_MC(dim:int, a:float|jax.Array, b:float|jax.Array, n:int, key:jax.Array):
    rr = jrnd.uniform(key, (n ** dim, dim), minval = a, maxval = b)
    ww = 1 / (n ** dim)
    return rr, ww

def mvrw_sobol(dim:int, a:float|jax.Array, b:float|jax.Array, n:int):
    """Sobol points and rectangular weights for a rectangular domain.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    a : array-like or scalar
        Lower bound(s) of the domain. Must be broadcastable to shape (dim,).
    b : array-like or scalar
        Upper bound(s) of the domain. Must be broadcastable to shape (dim,).
    n : int
        Number of subdivisions per dimension; total samples produced = n ** dim.

    Returns
    -------
    rr : jnp.ndarray, shape (n**dim, dim)
        Sobol points scaled to the hyper-rectangle [a, b].
    ww : jnp.ndarray, shape (n**dim,)
        Constant weights equal to volume_of_domain / (n ** dim).
    """
    # Straightforward
    rr = sobol(dim, n ** dim)
    rr = rr * (b - a) + a
    w = jnp.prod(b - a) / (n ** dim)
    ww = jnp.full(n ** dim, w)
    return rr, ww

def mvrw_GLQMC(dim:int, a:float|jax.Array, b:float|jax.Array, n:int, m:int = 50):
    """
    Compute multivariate quadrature nodes and product weights by interpolating
    1D Gauss–Legendre nodes/weights onto Sobol quasi-random coordinates and
    scaling to rectangular domain.

    Parameters
    ----------
    dim : int
        Number of dimensions (columns) for the multivariate points.
    a, b : float, jax.Array
        Interval endpoints used to scale nodes and weights. If jax.Array, must be
        of shape (dim,).
    n : int
        Number of Sobol points per dimension (total number produced: n ** dim).
    m : int
        Number of GL roots/weights for linear interpolation.

    Returns
    -------
    rr : jnp.ndarray
        Array of shape (n, dim) containing mapped abscissas in [a, b] for each
        Sobol sample and dimension.
    ww : jnp.ndarray
        1D array of length n containing the product (multivariate) weights
        corresponding to each row of rr, scaled for the interval [a, b].
    """
    # interpolation space
    i = jnp.linspace(-1., 1., m)
    # GL roots and weights for interpolation
    r_GL, w_GL = rw_P(m)
    # Sobol roots to stretch
    r_sobol, _ = mvrw_sobol(dim, -1., 1., n)
    # Treat data as (i, r_GL) and (i, w_GL) and interpolate values at sobol points
    rr = jax.vmap(jtu.Partial(jnp.interp, xp = i, fp = r_GL), in_axes = 1, out_axes = 1)(r_sobol)
    ww = jax.vmap(jtu.Partial(jnp.interp, xp = i, fp = w_GL), in_axes = 1, out_axes = 1)(r_sobol)
    # Stretch and scale
    rr = rr * (b - a) / 2 + (a + b) / 2
    ww = ww * (b - a) / 2
    # prod roots, scale for difference in point #
    ww = jnp.prod(ww, axis = 1) * (m / n)
    return rr, ww

# Multivariate Nystrom method
def HFIEQ2_nys(krnl, rr, ww):
    """
    Compute a Nyström-style eigendecomposition of a kernel function.
    ----------
    krnl : Callable
        Kernel function taking two scalar array arguments (r_i, r_j) and returning a scalar kernel value.
    rr : array_like
        1-D array of sample points (shape (N,)).
    ww : array_like
        1-D array of quadrature weights (shape (N,)); entries must be positive.
    Returns
    -------
    lmda : jnp.ndarray
        1-D array of eigenvalues of the weighted kernel matrix, sorted in descending order (shape (N,)).
    Phi : jnp.ndarray
        2-D array whose columns are the corresponding eigenvectors expressed in the original basis
        (shape (N, N)). The routine uses sqrt-weights to form a symmetric matrix for eigendecomposition
        and then rescales eigenvectors by 1/sqrt(ww) to return them to the unweighted basis.
    Notes
    -----
    - The kernel matrix K is formed as K_ij = krnl(rr[i], rr[j]).
    - To preserve symmetry the function uses D = diag(sqrt(ww)) and diagonalizes D K D.
    - ww must be strictly positive to permit sqrt and division by sqrt(ww).
    """
    # Root weights for symmetry
    ww_sqrt = jnp.sqrt(ww)

    # Diagonalize for matmul
    WW_sqrt = jnp.diag(ww_sqrt)

    # Kernel matrix
    K = jax.vmap(jax.vmap(krnl, in_axes = (None, 0)), in_axes = (0, None))(rr, rr)
    
    # Replace WW @ K_mat with WW_sqrt on either side to maintain symmetry
    B = WW_sqrt @ K @ WW_sqrt

    # Eigendecomp
    lmda, Phi = jnp.linalg.eigh(B)

    # Divide vectors by weight vector to get back to original basis
    lmda, Phi = lmda[::-1], Phi[:, ::-1] / ww_sqrt[:, None]

    return lmda, Phi