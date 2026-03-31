import jax
from jax import numpy as jnp 
from jax.numpy import linalg as jnpla

_eps = 1e-8

def C_Lacour(MlnR:jax.Array, Y:jax.Array, w:jax.Array):
    """
    Compute the covariance matrix using the kernel from Lacour & Abrahamson (2021) [DOI:10.1785/0120200381].

    Parameters
    ----------
    MlnR_all : jax.Array
        Array of shape (N, 2), where each row contains [M, lnR] pairs.
    y_all : jax.Array
        Array of values at input [M, lnR]. This argument is never used. Just kept so it works like the Paciorek kernel. 

    Returns
    -------
    jax.Array
        Correlation matrix of shape (N, N).
    """
    M, lnR = MlnR.T
    mu_Y = Y @ w
    std_Y = jnp.sqrt((Y - mu_Y[:, None])**2 @ w)
    crossvar_Y = std_Y[None] * std_Y[:, None]
    
    dM = jnp.abs(M[None] - M[:, None])
    dlnR = jnp.abs(lnR[None] - lnR[:, None])
    c1, c2, c3, c4, c5 = -0.1343, 0.0668, -0.4288, -0.0636, 0.0082
    # dlnR**2 updated from email exchange with Maxime; paper just says c4 * dlnR.
    corr = (c1 * dM + c2 * dlnR + jnp.exp(c3 * dM**2 + c4 * dlnR**2 + c5 * dM * dlnR))

    return crossvar_Y * corr

def C_svarg_iso(X:jax.Array, Y:jax.Array, w:jax.Array):
    """
    Guess the covariance kernel for input/output pair using the semivariogram of the inputs/outputs.

    Parameters
    ----------
    X : jax.Array
        Array of input locations (N, D).
    Y : jax.Array
        Array of values at input locations (N, D).

    Returns
    -------
    jax.Array
        Covariance matrix of shape (N, N).
    """
    # Dimension
    n, d_in = X.shape
    _, d_out = Y.shape

    # Scale X to [0, 1]
    x_a, x_b = X.min(axis = 0), X.max(axis = 0)
    X_scaled = (X - x_a) / (x_b - x_a)
    R_dim = jnp.abs(X_scaled[None] - X_scaled[:, None])
    R = jnpla.norm(R_dim, axis = -1)

    R_dim = jnp.abs(X[None] - X[:, None])

    # Calculate Y-features
    mu_Y = Y @ w # (n,)
    std_Y = jnp.sqrt((Y - mu_Y[:, None])**2 @ w) #(n,)
    crossvar_Y = std_Y[None] * std_Y[:, None]

    # Semivariogram will be important
    Svarg_Y = (Y[None] - Y[:, None])**2 @ w / 2 # (n, n)
    # Convert to correlation using cheap guess
    corr = (1 - Svarg_Y / Svarg_Y.max())# * (1 - R / R.max())**(1 / 2)
    return crossvar_Y * corr

def C_svarg_aniso(X:jax.Array, Y:jax.Array, w:jax.Array):
    """
    Guess the covariance kernel for input/output pair using the semivariogram of the inputs/outputs.

    Parameters
    ----------
    X : jax.Array
        Array of input locations (N, D).
    Y : jax.Array
        Array of values at input locations (N, D).

    Returns
    -------
    jax.Array
        Covariance matrix of shape (N, N).
    """
    # Dimension
    n, d_in = X.shape
    _, d_out = Y.shape

    # Scale X to [0, 1]
    x_a, x_b = X.min(axis = 0), X.max(axis = 0)
    X_scaled = (X - x_a) / (x_b - x_a)
    R_dim = jnp.abs(X_scaled[None] - X_scaled[:, None])
    R = jnpla.norm(R_dim, axis = -1)

    R_dim = jnp.abs(X[None] - X[:, None])

    # Calculate Y-features
    mu_Y = Y @ w # (n,)
    std_Y = jnp.sqrt((Y - mu_Y[:, None])**2 @ w) #(n,)
    crossvar_Y = std_Y[None] * std_Y[:, None]

    # Calculate semivariogram
    Svarg_Y = (Y[None] - Y[:, None])**2 @ w / 2 # (n, n)
    # Scale by fractional contributions of each dim to euclidean distance
    R_frac = R_dim**2 / R[:,:,None]**2
    R_frac = jnp.where(jnp.isnan(R_frac), 0, R_frac)
    Svarg_Y_proj = Svarg_Y[:,:,None] * R_frac

    # Convert to correlations using cheap guess and take product
    corr_proj = (1 - Svarg_Y_proj / Svarg_Y_proj.max(axis = (0, 1))) #* (1 - R_dim / R_dim.max(axis = (0, 1)))**(1 / 2)
    corr = jnp.prod(corr_proj, axis =-1)
    return crossvar_Y * corr

def C_Paciorek_iso(X:jax.Array, Y:jax.Array, w:jax.Array):
    """
    Compute the nonstationary isotropic covariance matrix using the kernel from Paciorek & Schervish (2003) [DOI:10.5555/2981345.2981380].

    Parameters
    ----------
    X : jax.Array
        Array of input locations (N, D).
    Y : jax.Array
        Array of values at input locations (N, D).

    Returns
    -------
    jax.Array
        Covariance matrix of shape (N, N).
    """
    # Dimension
    n, d_in = X.shape
    _, d_out = Y.shape

    # Scale X to [0, 1]
    x_a, x_b = X.min(axis = 0), X.max(axis = 0)
    X_scaled = (X - x_a) / (x_b - x_a)
    R = jnpla.norm(X_scaled[None] - X_scaled[:, None], axis = -1)

    # Calculate Y-features
    mu_Y = Y @ w # (n,)
    std_Y = jnp.sqrt((Y - mu_Y[:, None])**2 @ w) #(n,)
    crossvar_Y = std_Y[None] * std_Y[:, None]
    # Semivariogram will be important
    Svarg_Y = (Y[None] - Y[:, None])**2 @ w / 2 # (n, n)
    # Global differences
    svarg_Y = Svarg_Y.sum(axis = -1)
    svarg_Y = svarg_Y / jnp.min(svarg_Y)

    # INVERSE OF svarg_Y.
    l = 1 / svarg_Y
    prodterm = (l[None] * l[:, None])**(1 / 4)
    coeffterm = (2 / (l[None] + l[:, None])) ** (1 / 2)
    corr = prodterm * coeffterm * jnp.exp(-(R * coeffterm)**2)
    return crossvar_Y * corr

def C_Paciorek_aniso(X:jax.Array, Y:jax.Array, w:jax.Array):
    """
    Compute the nonstationary anisotropic covariance matrix using the kernel from Paciorek & Schervish (2003) [DOI:10.5555/2981345.2981380].

    Parameters
    ----------
    X : jax.Array
        Array of input locations (N, D).
    Y : jax.Array
        Array of values at input locations (N, D).

    Returns
    -------
    jax.Array
        Covariance matrix of shape (N, N).
    """
    # Dimension
    n, d_in = X.shape
    _, d_out = Y.shape

    # Scale X to [0, 1]
    x_a, x_b = X.min(axis = 0), X.max(axis = 0)
    X_scaled = (X - x_a) / (x_b - x_a)
    R_dim = jnp.abs(X_scaled[None] - X_scaled[:, None])
    R = jnpla.norm(R_dim, axis = -1)

    # Calculate Y-features
    mu_Y = Y @ w # (n,)
    std_Y = jnp.sqrt((Y - mu_Y[:, None])**2 @ w) #(n,)
    crossvar_Y = std_Y[None] * std_Y[:, None]
    
    # Calculate semivariogram
    Svarg_Y = (Y[None] - Y[:, None])**2 @ w / 2 # (n, n)
    # Scale by fractional contributions of each dim to euclidean distance
    R_frac = R_dim**2 / R[:,:,None]**2
    R_frac = jnp.where(jnp.isnan(R_frac), 0, R_frac)
    Svarg_Y_Xproj = Svarg_Y[:,:,None] * R_frac
    svarg_Y_Xproj = Svarg_Y_Xproj.sum(axis = 1) 
    svarg_Y_Xproj = svarg_Y_Xproj / svarg_Y_Xproj.min(axis = 0)

    # Inverse of projected semivariogram
    l = 1 / svarg_Y_Xproj
    prodterm = (l[None] * l[:, None])**(1 / 4)
    coeffterm = (2 / (l[None] + l[:, None])) ** (1 / 2)
    corrs = prodterm * coeffterm * jnp.exp(-(R[:,:,None] * coeffterm)**2)
    C = crossvar_Y * jnp.prod(corrs, axis = -1)
    return C

def C_fullcorr(X:jax.Array, Y:jax.Array, w:jax.Array):
    """
    Compute the nonstationary anisotropic covariance matrix using the kernel from Paciorek & Schervish (2003) [DOI:10.5555/2981345.2981380].

    Parameters
    ----------
    X : jax.Array
        Array of input locations (N, D).
    Y : jax.Array
        Array of values at input locations (N, D).

    Returns
    -------
    jax.Array
        Covariance matrix of shape (N, N).
    """
    # Dimension
    n, d_in = X.shape
    _, d_out = Y.shape

    # Calculate Y-features
    mu_Y = Y @ w # (n,)
    std_Y = jnp.sqrt((Y - mu_Y[:, None])**2 @ w) #(n,)
    crossvar_Y = std_Y[None] * std_Y[:, None]
    
    return crossvar_Y