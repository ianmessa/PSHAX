import jax
from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from numerics.spectral import mv_psi

_eps = 1e-14

def _phs(x:jax.Array, c:jax.Array, r:float, k:int):
    """
    Computes the polyharmonic spline value r^k * log(r).
    
    Parameters
    ----------
    x : jax.Array
        Point of shape (d,).
    c : jax.Array
        Center of shape (d,).
    R : float
        Precomputed Euclidean distance ||x - c||.
    k : int, optional
        Order of the spline (default is 2).
    
    Returns
    -------
    jax.Array
        Spline value.
    """
    # Easy
    r_safe = jnp.where(jnp.abs(r) < _eps, 1., r)
    val = jnp.where(k % 2 == 0, jnp.log(r_safe), 1.)
    return r ** k * val

def rbf_interpolator(x:jax.Array, y:jax.Array, k_phs:int, k_poly:int):
    """
    Build an RBF interpolant using a polyharmonic spline (PHS) with a
    polynomial tail.

    Constructs and solves the block system
        [Phi_phs   P] [w]   = [y]
        [P^T      0] [lam]   [0]
    where Phi_phs is the PHS matrix evaluated on x and P is the polynomial
    basis matrix (from mv_psi). Returns a callable that evaluates the
    interpolant at new points.

    Parameters
    ----------
    x : jax.Array
        Array of interpolation centers with shape (n, d).
    y : jax.Array
        Values at centers with shape (n,).
    k_phs : int
        Order of the polyharmonic spline.
    k_poly : int
        Polynomial degree / basis parameter for mv_psi.

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        A function rbf_interpolate(xi) that evaluates the interpolant at
        query points xi (expected shape (m, d)) and returns an array
        of interpolated values of shape (m,).
    """
    # Get dims
    d = x.shape[-1]
    # Build PHS matrix
    R = jnpla.norm(x[:, None] - x[None], axis = -1)
    phs_map = jax.vmap(jax.vmap(_phs, in_axes = (None, 0, 0, None)), in_axes = (0, None, 0, None))
    Phi = phs_map(x, x, R, k_phs)
    # Build polynomial block + zero block in lower right
    alpha_poly = jnp.zeros(d)
    poly = mv_psi(x, 'M', k_poly, alpha_poly)
    dk_poly = poly.shape[-1]
    zeros = jnp.zeros((dk_poly, dk_poly))
    # Put it all together
    Phi = jnp.block([[Phi, poly], 
                     [poly.T, zeros]])
    # Solve for y + zeros at tail
    y_aug = jnp.concat([y, jnp.zeros(dk_poly)])
    w_all = jnpla.solve(Phi, y_aug)
    w, lmda = w_all[:-dk_poly], w_all[-dk_poly:]

    def rbf_interpolate(xi):
        """
        Evaluate the RBF interpolant at query points.

        Parameters
        ----------
        xi : jax.Array
            Query points with shape (m, d).

        Returns
        -------
        jax.Array
            Interpolated values at xi with shape (m,).
        """
        # Similar routine to above
        Ri = jnpla.norm(xi[:, None] - x[None], axis = -1)
        Phii = phs_map(xi, x, Ri, k_phs)
        polyi = mv_psi(xi, 'M', k_poly, alpha_poly)
        return Phii @ w + polyi @ lmda

    return rbf_interpolate