from jax import lax
from jax.tree_util import Partial
from jax import numpy as jnp

from gm_utils import *

def f_epistemic(gmm, factor:float, scn:gm_scenario):
    T, lnSA, sigma = gmm(scn)
    sig_u = 0.083 + 0.056 * jnp.clip(scn.Mw - 7, min = 0)
    T_factor = lax.select(T >= 1.0, 0.171 * jnp.log(T), jnp.zeros_like(T))
    SOF_factor = lax.select(scn.SOF_flag == 1, 0.038, 0.)
    delta_lnSA = factor * (sig_u + T_factor + SOF_factor)
    return T, lnSA + delta_lnSA, sigma

def f_epistemic_AAY14(gmm, factor):
    return Partial(f_epistemic, gmm, factor)