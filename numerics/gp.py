import jax
from jax import numpy as jnp 
from jax.numpy import linalg as jnpla

def C_Lacour(MlnR_all:jax.Array):
    """
    Compute the correlation matrix using the kernel from Lacour & Abrahamson (2021) [DOI:10.1785/0120200381].

    Parameters
    ----------
    MlnR_all : jax.Array
        Array of shape (N, 2), where each row contains [M, lnR] pairs.

    Returns
    -------
    jax.Array
        Correlation matrix of shape (N, N).
    """
    M, lnR = MlnR_all.T
    dM = jnp.abs(M[None] - M[:, None])
    dlnR = jnp.abs(lnR[None] - lnR[:, None])
    c1, c2, c3, c4, c5 = -0.1343, 0.0668, -0.4288, 0.0636, 0.0082
    return c1 * dM + c2 * dlnR + jnp.exp(c3 * dM**2 + c4 * dlnR + c5 * dM * dlnR)

def C_Paciorek(x_all:jax.Array, y_all:jax.Array):
    """
    Compute the nonstationary covariance matrix using the kernel from Paciorek & Schervish (2003) [DOI:10.5555/2981345.2981380].

    Parameters
    ----------
    x_all : jax.Array
        Array of input locations (N, D).
    y_all : jax.Array
        Array of values at input locations (N, D).

    Returns
    -------
    jax.Array
        Covariance matrix of shape (N, N).
    """
    # Dimension
    d = y_all.shape[-1]
    # Standard deviation
    std = y_all.std(axis = -1, ddof = 1)
    # Crossvariance
    crossvar = std[None] * std[:, None]
    # Covariogram
    covarg = jnpla.norm(x_all[None] - x_all[:, None], axis = -1) / d

    # Lengthscale + adaptive kernel from Paciorek + Schervish
    lscale = 1 / (std + 1e-4)
    lscale_prod = lscale[None] * lscale[:, None]
    lscale_sqsum = lscale[None]**2 + lscale[:, None]**2
    C = crossvar * ((2 * lscale_prod) / lscale_sqsum) ** (1 / 2) * jnp.exp(-covarg**2 / lscale_sqsum)
    return C