import jax
from jax import lax
from jax.tree_util import Partial
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .gm_utils import *

##### GROUND MOTION COEFFICIENTS #####
gmc = pl.read_csv(files("seismic") / "ASK14_coeffs.csv")
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.

gmc = gmc.with_columns([pl.col("T").cast(pl.Float64)])
gmc = gmc.sort('T')
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

# First few
T_ASK, v_lin_all, b_all = gmc[:3]
N_all, M1_all, c_all, c4_all = gmc[3:7]
M2 = 5.0
# Update M with constant for M2 and 0-row for M0
empty_all = jnp.zeros_like(T_ASK)

# Get a. But they're all out of order...
a_all = gmc[7:38]
a_idcs = jnp.array([int(ai[1:]) for ai in gmc_col[7:38]])
a_all = a_all[jnp.argsort(a_idcs)]
a_missing = jnp.arange(a_idcs.max())
a_missing = a_missing[~jnp.isin(a_missing, a_idcs)]
# Update indices to account for insertions
a_missing = a_missing - jnp.arange(a_missing.shape[0])
a_all = jnp.insert(a_all, a_missing, empty_all, axis = 0)

# Grab s
s_est_all = gmc[38:40]
s_est_all = jnp.insert(s_est_all, 0, empty_all, axis = 0)
s_all = gmc[40:]
s_all = jnp.insert(s_all, 0, empty_all, axis = 0)

A = 610 ** 4
B = 1360 ** 4 + A
vs_rock = 1180
a2_HW = 0.2
H1, H2, H3 = 0.25, 1.5, -0.75
phi_amp_sq = 0.16
# Omit 1000 because of Jax interp behavior (constant extrapolation for
#   values above 700 so we don't need to worry about 1000)
vs30_bins = jnp.array([150., 250., 400., 700.])

# Also defining v1 so we don't need to do it twice
v1_all = jnp.exp(-0.35 * jnp.log(T_ASK / 0.5) + jnp.log(1500))
v1_all = v1_all.at[T_ASK <= 0.5].set(1500)
v1_all = v1_all.at[T_ASK >= 3].set(800)

c_other_all = jnp.stack([M1_all, b_all, c_all, c4_all, v_lin_all, v1_all, N_all], axis = 0)

def slice_coeffs(T):
    T_idx = jnp.searchsorted(T_ASK, T) - 1
    T_slice = lax.dynamic_slice_in_dim(T_ASK, T_idx, 2, axis = -1)
    a = lax.dynamic_slice_in_dim(a_all, T_idx, 2, axis = -1)
    s_est = lax.dynamic_slice_in_dim(s_est_all, T_idx, 2, axis = -1)
    s = lax.dynamic_slice_in_dim(s_all, T_idx, 2, axis = -1)
    c_other = lax.dynamic_slice_in_dim(c_other_all, T_idx, 2, axis = -1)
    return (T_slice, a, c_other, (s_est, s))

def f_dAmp(vs30, SA_rock,
           c_other):
    _, b, c, _, v_lin, _, N = c_other
    dAmp = (-b * SA_rock) / (SA_rock + c) + (b * SA_rock) / (SA_rock + c * (vs30 / v_lin) ** N)
    return jnp.where(vs30 >= v_lin, 0., dAmp)

def f1(Mw, R_rup, R, a,
       M1):
    f1_base = a[1] + a[17] * R_rup
    M_floor = jnp.clip(Mw, min = M2)
    dM1 = M_floor - M1
    dM_max = (8.5 - M_floor) ** 2
    dM2 = Mw - M2
    
    A_capped = a[4] + (a[5] - a[4]) * (Mw > M1)
    dM2_coeff = (Mw < M2) * (a[6] * dM2)

    return f1_base + A_capped * dM1 + a[8] * dM_max + dM2_coeff * dM2 + jnp.log(R) * (a[2] + a[3] * dM1)

def f7_8(Mw, SOF_flag,
         a):
    lnSA_SOF = a[12] * jnp.clip(Mw - 4, min = 0, max = 1)
    return jnp.where(SOF_flag == 1., lnSA_SOF, 0.)

def f5(vs30, SA_rock,
       a, c_other):
    _, b, c, _, v_lin, v1, N = c_other
    vs30_star = jnp.clip(vs30, max = v1)
    return a[10] * jnp.log(vs30_star / v_lin) - b * jnp.log(SA_rock + c) + b * jnp.log(SA_rock + c * (vs30_star / v_lin) ** N)

def f4(Mw, width, dip, R_jb, R_x, z_tor,
       a):
    T1 = jnp.clip(2 - (dip / 45), min = 4 / 3)

    T2 = 1 + a2_HW * (Mw - 6.5) + jnp.where(Mw < 6.5, (1 - a2_HW) * (Mw - 6.5) ** 2, 0.)
    
    r1 = width * jnp.cos(jnp.deg2rad(dip))
    r2 = 3 * r1
    Rxr1 = R_x / r1
    leq_r1 = H1 + H2 * Rxr1 + H3 * Rxr1 ** 2
    mask_r1 = R_x <= r1

    leq_r2 = 1 - (R_x - r1) / (r2 - r1)
    mask_r2 = (r1 < R_x) & (R_x <= r2)

    T3 = 0. + leq_r1 * mask_r1 + leq_r2 * mask_r2

    T4 = 1 - z_tor ** 2 / 100

    T5 = 1 - R_jb / 30

    return a[13] * T1 * T2 * T3 * T4 * T5

def f6(z_tor, 
       a):
    return a[15] * jnp.clip(z_tor / 20, max = 1)

def f10(vs30, z1p0,
        a):
    z1p0_ref = jnp.exp(-7.67 / 4.0 * jnp.log((vs30 ** 4 + A) / B)) / 1000
    z1p0_soil = jax.vmap(jnp.interp, in_axes = (None, None, 1))(vs30, vs30_bins, a[43:47])
    z1p0_soil = z1p0_soil * jnp.log((z1p0 + 0.1) / (z1p0_ref + 0.1))
    return z1p0_soil

def f_lnSA_SA_rock(Mw, width, dip, z_tor, SOF_flag,
                   vs30, z1p0,
                   R_jb, R_rup, R_x,
                   a, c_other
                   ):
    _, b, _, c4, v_lin, v1, N = c_other
    c4_mag = jnp.clip(c4 - (c4 - 1) * (5 - Mw), min = 1, max = c4)
    R = (R_rup ** 2 + c4_mag ** 2) ** (1 / 2)
    vs_rock_capped = jnp.clip(vs_rock, max = v1)
    lnSA5_rock = (a[10] + b * N) * jnp.log(vs_rock_capped / v_lin)
    
    lnSA1 = f1(Mw, R_rup, R, a, c_other[0])
    lnSA7_8 = f7_8(Mw, SOF_flag, a)
    lnSA4 = f4(Mw, width, dip, R_jb, R_x, z_tor, a)
    lnSA6 = f6(z_tor, a)
    lnSA10 = f10(vs30, z1p0, a)

    SA_rock = jnp.exp(lnSA1 + lnSA7_8 + lnSA5_rock + lnSA4 + lnSA6)

    lnSA5 = f5(vs30, SA_rock, a, c_other)

    lnSA = lnSA1 + lnSA7_8 + lnSA5 + lnSA4 + lnSA6 + lnSA10

    return lnSA, SA_rock

def f_sigma(Mw, vs30, vs30inf_flag, SA_rock,
            sigma_coeffs, c_other):
    s_est, s = sigma_coeffs
    dAmp_p1 = f_dAmp(vs30, SA_rock, c_other) + 1
    vs30_s = jnp.where(vs30inf_flag == 1., s_est, s[:3])

    phi_A = jnp.clip(vs30_s[1] + ((vs30_s[2] - vs30_s[1]) / 2) * (Mw - 4.0), min = s[1], max = s[2])
    phi_B_sq = phi_A ** 2 - phi_amp_sq
    phi_sq = phi_B_sq * dAmp_p1 ** 2 + phi_amp_sq

    tau_B = jnp.clip(s[3] + ((s[4] - s[3]) / 2) * (Mw - 4.0), min = s[3], max = s[4])
    tau = tau_B * dAmp_p1

    return (phi_sq + tau ** 2) ** (1 / 2)

def f_ASK14(Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
    T_slice, a, c_other, sigma_coeffs = slice_coeffs(T) 

    SOF_flag = fault.calc_SOF_flag()
    R_jb, R_rup, R_epi, R_hyp, R_x = R
    
    lnSA, SA_rock = f_lnSA_SA_rock(Mw, fault.width, fault.dip, fault.z_tor, SOF_flag,
                                   site.vs30, site.z1p0, R_jb, R_rup, R_x,
                                   a, c_other)
    std = f_sigma(Mw, site.vs30, site.vs30inf_flag, SA_rock,
                  sigma_coeffs, c_other)
    lnSA = jnp.interp(T, T_slice, lnSA)
    std = jnp.interp(T, T_slice, std)
    return lnSA, std