# Jaxlib
import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .gm_utils import *

gmc = pl.read_csv(files("seismic") / "Idriss14_coeffs.csv")
gmc[-1, 'T'] = -1.
gmc = gmc.with_columns([pl.col("T").cast(pl.Float64)])
gmc = gmc.sort('T')
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T_Idriss = gmc[0]
T_sort = jnp.argsort(T_Idriss)
empty_all = jnp.zeros_like(T_Idriss, dtype = float)

# a
a_lo_all = jnp.concat([empty_all[None], gmc[1:3], gmc[-4][None]], axis = 0)
a_hi_all = jnp.concat([empty_all[None], gmc[5:7], gmc[-4][None]], axis = 0)
a_all = jnp.stack([a_lo_all, a_hi_all], axis = 1)

# b
b_lo_all = jnp.concat([empty_all[None], gmc[3:5]], axis = 0)
b_hi_all = jnp.concat([empty_all[None], gmc[7:9]], axis = 0)
b_all = jnp.stack([b_lo_all, b_hi_all], axis = 1)

# Other coeffs
xi_all, gamma_all, phi_all = gmc[-3:]

def slice_coeffs(T):
    T_idx = jnp.searchsorted(T_Idriss, T) - 1
    T_slice = lax.dynamic_slice_in_dim(T_Idriss, T_idx, 2, axis = -1)
    a = lax.dynamic_slice_in_dim(a_all, T_idx, 2, axis = -1)
    b = lax.dynamic_slice_in_dim(b_all, T_idx, 2, axis = -1)
    xi = lax.dynamic_slice_in_dim(xi_all, T_idx, 2, axis = -1)
    gamma = lax.dynamic_slice_in_dim(gamma_all, T_idx, 2, axis = -1)
    phi = lax.dynamic_slice_in_dim(phi_all, T_idx, 2, axis = -1)
    return T_slice,a,b,xi,gamma,phi

def f_lnSA(Mw, R_rup, vs30, RV_flag, a, b, xi, gamma, phi):
    # Select coeffs based on magnitude
    a_sel = jnp.where(Mw >= 6.75, a[:, 1], a[:, 0])
    b_sel = jnp.where(Mw >= 6.75, b[:, 1], b[:, 0])
    # Done
    return a_sel[1] + \
           a_sel[2] * Mw + \
           a_sel[3] * (8.5 - Mw) ** 2 - \
           (b_sel[1] + b_sel[2] * Mw) * jnp.log(R_rup + 10) + \
           xi * jnp.log(vs30) + gamma * R_rup + phi * RV_flag

def f_sigma(Mw, T):
    return 1.18 + 0.035 * jnp.log(T) - 0.06 * Mw

def f_Idriss14(Mw:float, T:float, site:Site, fault:Fault, R:jax.Array): 
    T_slice, a, b, xi, gamma, phi = slice_coeffs(T)

    SOF_flag = fault.calc_SOF_flag()
    RV_flag = SOF_flag < 0
    R_jb, R_rup, R_epi, R_hyp, R_x = R
    
    lnSA = f_lnSA(Mw, R_rup, site.vs30, RV_flag, a, b, xi, gamma, phi)
    std = f_sigma(Mw, T_slice)
    lnSA = jnp.interp(T, T_slice, lnSA)
    std = jnp.interp(T, T_slice, std)
    return lnSA, std