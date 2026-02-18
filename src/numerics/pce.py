import jax
from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax.experimental.jet import jet
from jax.scipy.special import factorial

from numerics.spectral import mv_psi
from collections.abc import Callable

def PCE(x:jax.Array, y:jax.Array, 
    k_PCE:int = 5, strategy:str = 'hyperbolic', q:float = 0.625, 
    alpha:float|jax.Array = 1.):
    """
    Fit a Polynomial Chaos Expansion (PCE) to data using least squares.

    Parameters
    ----------
    x : jax.Array
    Input data of shape (n_samples, n_features).
    y : jax.Array
    Target values of shape (n_samples,) or (n_samples, n_targets).
    k_PCE : int, optional
    Maximum polynomial order (default is 5).
    strategy : str, optional
    Polynomial basis strategy, e.g., 'hyperbolic' (default).
    q : float, optional
    Hyperbolic truncation parameter (default is 0.625).
    alpha : float or jax.Array, optional
    Scaling parameter(s) for the basis (default is 1.).

    Returns
    -------
    eval_PCE : Callable
    Function that evaluates the fitted PCE at new input locations.
    """
    # Just use least squares to fit our data
    H = mv_psi(x, 'H', k_PCE, strategy, q, alpha)
    c = jnpla.lstsq(H, y)[0]

    def eval_PCE(xi):
        """
        Evaluate the fitted PCE at new input locations.

        Parameters
        ----------
        xi : jax.Array
            New input data of shape (n_eval, n_features).

        Returns
        -------
        y_pred : jax.Array
            Predicted values from the PCE.
        """
        Hi = mv_psi(xi, 'H', k_PCE, strategy, q, alpha)
        return Hi @ c
    return eval_PCE

def tPCE(x:jax.Array, f:Callable, z0:float, 
           k_taylor:int, 
           k_PCE:int = 5, strategy:str = 'hyperbolic', q:float = 0.625, 
           alpha:float|jax.Array = 1.):
    """
    Construct a Taylor-Polynomial Chaos Expansion (tPCE) in variable z for a function f(z, x) using least squares.

    Parameters
    ----------
    x : jax.Array
    Input data for x of shape (n_samples, n_features).
    f : Callable
    Function of the form f(z, x).
    z0 : float
    Expansion point for the Taylor series in z.
    k_taylor : int
    Order of the Taylor expansion in z.
    k_PCE : int, optional
    Maximum polynomial order for PCE in x (default is 5).
    strategy : str, optional
    Polynomial basis strategy (default is 'hyperbolic').
    q : float, optional
    Hyperbolic truncation parameter (default is 0.625).
    alpha : float or jax.Array, optional
    Scaling parameter(s) for the basis (default is 1.).

    Returns
    -------
    eval_tPCE : Callable
    Function that evaluates the tPCE at new (z, x) pairs.
    """
    # Define f only in terms of z
    fz = lambda z: f(z, x)
    # Hermite polynomials
    H = mv_psi(x, 'H', k_PCE, strategy, q, alpha)
    # Take derivatives of fz
    y_z0, dnydzn_z0 = jet(fz, (z0,), ((1,) + (0,) * (k_taylor - 1),))
    dnydzn_z0 = jnp.stack([y_z0, *dnydzn_z0])
    # Use least-squares to get Taylor series coefficient derivatives by chain rule
    #   (lstsq is a linear operator) at z0
    c = jnpla.lstsq(H, dnydzn_z0.T)[0]

    def eval_c(z:float):
        """
        Evaluate the Taylor coefficients at a given z.

        Parameters
        ----------
        z : float
            Value of z at which to evaluate the Taylor expansion.

        Returns
        -------
        c_z : jax.Array
            Coefficient vector for the expansion at z.
        """
        # Monomials for taylor expansion
        zM = uv_psi(z - z0, 'M', k_taylor + 1)
        # Factorial denominator
        denom = factorial(jnp.arange(k_taylor + 1))
        # Bam
        return c @ (zM / denom)
    
    def eval_tPCE(zi:float, xi:jax.Array):
        """
        Evaluate the tPCE at a given (z, x) pair.

        Parameters
        ----------
        zi : float
            Value of z.
        xi : jax.Array
            Value(s) of x (shape (n_features,) or (n_eval, n_features)).

        Returns
        -------
        y_pred : jax.Array
            Predicted value(s) from the tPCE.
        """
        Hi = mv_psi(xi, 'H', k_PCE, strategy, q, alpha)
        ci = eval_c(zi)
        return Hi @ ci
    return eval_tPCE