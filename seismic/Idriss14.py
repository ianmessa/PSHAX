# Jaxlib
import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .gm_utils import *

gmc = pl.read_csv(files("seismic") / "Idriss14_coeffs.csv")
gmc[-1, 'T'] = -1.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T = gmc[0]
T_sort = jnp.argsort(T)
empty = jnp.zeros_like(T, dtype = float)

# a
a = jnp.empty((4, 2, T.shape[0]))
# a1_lo, a2_lo
a = a.at[1:3, 0].set(gmc[1:3])
# a1_hi, a2_hi
a = a.at[1:3, 1].set(gmc[5:7])
# a3
a = a.at[3, :].set(gmc[-4])

# b
b = jnp.empty((3, 2, T.shape[0]))
# b1_lo, b2_lo
b = b.at[1:, 0].set(gmc[3:5])
# b1_hi, b2_hi
b = b.at[1:, 1].set(gmc[7:9])

# Other coeffs
xi, gamma, phi = gmc[-3:]

def f_lnSA(Mw, R_rup, vs30, RV_flag):
    # Select coeffs based on magnitude
    a_sel = jnp.where(Mw >= 6.75, a[:, 1], a[:, 0])
    b_sel = jnp.where(Mw >= 6.75, b[:, 1], b[:, 0])
    # Done
    return a_sel[1] + \
           a_sel[2] * Mw + \
           a_sel[3] * (8.5 - Mw) ** 2 - \
           (b_sel[1] + b_sel[2] * Mw) * jnp.log(R_rup + 10) + \
           xi * jnp.log(vs30) + gamma * R_rup + phi * RV_flag

def f_sigma(Mw):
    return 1.18 + 0.035 * jnp.log(T) - 0.06 * Mw

def f_Idriss14(Mw:float, site:Site, fault:Fault): 
    SOF_flag = fault.calc_SOF_flag()
    RV_flag = SOF_flag < 0
    R_rup = calc_R_rup(site, fault)
    lnSA = f_lnSA(Mw, R_rup, site.vs30, RV_flag)
    std = f_sigma(Mw)
    lnSA = jnp.interp(T_master, T[T_sort], lnSA[T_sort])
    std = jnp.interp(T_master, T[T_sort], std[T_sort])
    return lnSA, std