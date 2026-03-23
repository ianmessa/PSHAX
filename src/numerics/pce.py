import jax
from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax import tree_util as jtu
from jax.experimental.jet import jet
from jax.scipy.special import factorial
from jax.scipy.stats.multivariate_normal import pdf as mv_norm_pdf

from numerics.spectral import uv_psi, mv_psi
from collections.abc import Callable

from matplotlib import pyplot as plt

@jtu.register_pytree_node_class
class PCE:
    def __init__(self, f:Callable,
                 k_PCE:int = 5, 
                 strategy:str = 'hyperbolic', 
                 q:float = 0.625, order_matters:bool = True, 
                 alpha:float|jax.Array = 1.,):
        self.f = f
        self.k_PCE = k_PCE
        self.strategy = strategy
        self. q = q
        self.order_matters = order_matters
        self.alpha = alpha
    
    def calc_coeffs(self, x:jax.Array, *f_args):
        def fx(xi):
            return self.f(xi, *f_args)
        d = x.shape[-1]
        y = jax.vmap(fx)(x)
        H = mv_psi(x, 'H', 
                   self.k_PCE, self.strategy, 
                   self.q, self.order_matters, 
                   self.alpha)
        H_norm2 = jnp.vecdot(H, H, axis = 0)
        c = (H.T @ y) / H_norm2
        return c
    
    def eval_coeffs(self, xi:jax.Array, c:jax.Array):
        Hi = mv_psi(xi, 'H', 
                   self.k_PCE, self.strategy, 
                   self.q, self.order_matters, 
                   self.alpha)
        return Hi @ c
    
    def tree_flatten(self):
        leaves = (self.alpha,)
        aux = (self.f, self.k_PCE, 
               self.strategy, self.q, self.order_matters)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        f, k_PCE, strategy, q, order_matters = aux
        (alpha,) = leaves
        return cls(f,
                   k_PCE=k_PCE, strategy=strategy,
                   q=q, order_matters=order_matters,
                   alpha=alpha)


@jtu.register_pytree_node_class
class tPCE:
    def __init__(self, f:Callable, 
                 k_taylor:int = 5, k_PCE:int = 5, 
                 strategy:str = 'hyperbolic',
                 q:float = 0.625, order_matters:bool = True,
                 alpha:float|jax.Array = 1.):
        self.f = f
        self.k_taylor = k_taylor
        self.k_PCE = k_PCE
        self.strategy = strategy
        self. q = q
        self.order_matters = order_matters
        self.alpha = alpha
    
    def calc_coeffs(self, z0:float, x:jax.Array, *f_args):
        def one_jet(xi, *args):
                def fz(zi):
                    return self.f(zi, xi, *args)
                return jet(fz, (z0,), ((1,) + (0,) * (self.k_taylor - 1),))

        # Hermite polynomials
        H = mv_psi(x, 'H', 
                   self.k_PCE, self.strategy, 
                   self.q, self.order_matters, 
                   self.alpha)

        # Take derivatives of fz
        y_z0, dnydzn_z0 = jax.vmap(one_jet, in_axes = (0,) + (None,) * len(f_args))(x, *f_args)
        dnydzn_z0 = jnp.stack([y_z0, *dnydzn_z0])

        # Use least-squares to get Taylor series coefficient derivatives by chain rule
        #   (lstsq is a linear operator) at z0
        H_norm2 = jnp.vecdot(H, H, axis = 0)
        c = (H.T @ dnydzn_z0.T) / H_norm2[:, None]

        # Evaluate taylor coefficients
        def eval_c(z:float):
            """
            Evaluate the tPCE coefficient array at a given deterministic input z.

            Combines the least-squares PCE coefficient matrix c with a monomial
            Taylor basis evaluated at (z - z0), weighted by factorial denominators,
            to return the effective PCE coefficients for the scalar value z.

            Args:
                z: Scalar deterministic input at which to evaluate the Taylor expansion.

            Returns:
                Array of shape (P, ...) containing the PCE coefficients at z.
            """
            # Monomials for taylor expansion
            transf_zM = uv_psi(z - z0, 'M', self.k_taylor + 1)
            # Factorial denominator
            denom = factorial(jnp.arange(self.k_taylor + 1))
            # Bam
            return c @ (transf_zM / denom)

        return eval_c
    
    def eval_coeffs(self, zi:float, xi:jax.Array, eval_c:Callable):
        Hi = mv_psi(xi, 'H', 
                   self.k_PCE, self.strategy, 
                   self.q, self.order_matters, 
                   self.alpha)
        c = eval_c(zi)
        return Hi @ c
    
    def tree_flatten(self):
        leaves = (self.alpha,)
        aux = (self.f, self.k_taylor, self.k_PCE,
               self.strategy, self.q, self.order_matters)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        f, k_taylor, k_PCE, strategy, q, order_matters = aux
        (alpha,) = leaves
        return cls(f, 
                   k_taylor=k_taylor, k_PCE=k_PCE, strategy=strategy,
                   q=q, order_matters=order_matters,
                   alpha=alpha)