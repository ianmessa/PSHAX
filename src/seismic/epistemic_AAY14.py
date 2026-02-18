from jax import lax
from jax.tree_util import Partial
from jax import numpy as jnp

from .gm_utils import *

def f_epistemic(gmm, factor:float, Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
    lnSA, std = gmm(Mw, T, site, fault, R)
    SOF_flag = fault.calc_SOF_flag()
    sig_u = 0.083 + 0.056 * jnp.clip(Mw - 7, min = 0)
    T_factor = lax.select(T >= 1.0, 0.171 * jnp.log(T), T * 0)
    SOF_factor = lax.select(SOF_flag == 1, 0.038, 0.)
    delta_lnSA = factor * (sig_u + T_factor + SOF_factor)
    return lnSA + delta_lnSA, std

def f_epistemic_AAY14(gmm, factor):
    return Partial(f_epistemic, gmm, factor)