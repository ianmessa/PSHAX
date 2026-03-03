import jax
from jax import lax
from jax.tree_util import Partial
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .seismic_utils import *

##### GROUND MOTION COEFFICIENTS #####
gmc = pl.read_csv(files("seismic") / "ASK14_coeffs.csv")
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.

gmc = gmc.with_columns([pl.col("T").cast(pl.Float64)])
gmc = gmc.sort('T')
gmc_col = gmc.columns
gmcdf = gmc
gmc = gmc.cast(pl.Float64).to_jax().T

T_ASK, v_lin_all, b_all, _, M1_all, c_all, c4_all = gmc[:7]
empty = jnp.zeros_like(T_ASK)
a1thru8_all = gmc[7:15]
a10thru17_all = gmc[15:23]
a43thru46_all = gmc[23:27]
a25_all, a28_all, a29_all, a31_all = gmc[27:31]
a36thru42_all = gmc[31:38]

a_all = jnp.concatenate([empty[None], a1thru8_all, empty[None], a10thru17_all] + \
                         7 * [empty[None]] + [a25_all[None]] + \
                         2 * [empty[None]] + \
                             [a28_all[None], a29_all[None], empty[None], a31_all[None]] + \
                         4 * [empty[None]] + \
                             [a36thru42_all, a43thru46_all])

s1est_all, s2est_all, s3_all, s4_all, s1m_all, s2m_all, s5_all, s6_all = gmc[38:]
s_all = jnp.stack([empty, s1m_all, s2m_all, 
                         s3_all, s4_all, 
                         s5_all, s6_all])
s_est_all = jnp.stack([empty, s1est_all, s2est_all])

M2 = 5.0
A = 610 ** 4
B = 1360 ** 4 + A
vs_rock = 1180
A2_HW = 0.2
A3, A4, A5 = 0.275, -0.1, -0.41
N = 1.5
H1, H2, H3 = 0.25, 1.5, -0.75
c4 = 4.5
phi_amp_sq = 0.16
# Omit 1000 because of Jax interp behavior (constant extrapolation for
#   values above 700 so we don't need to worry about 1000)
vs30_bins = jnp.array([150., 250., 400., 700.])

# Also defining v1 so we don't need to do it twice
v1_all = jnp.exp(-0.35 * jnp.log(T_ASK / 0.5) + jnp.log(1500))
v1_all = v1_all.at[T_ASK <= 0.5].set(1500)
v1_all = v1_all.at[T_ASK >= 3].set(800)

other_all = jnp.stack([M1_all, b_all, c_all, c4_all, v_lin_all, v1_all])

coeffs_all = jnp.concat([a_all, s_all, s_est_all, other_all])

def slice_coeffs(T):
    T_idx = jnp.searchsorted(T_ASK, T) - 1
    T_slice = lax.dynamic_slice_in_dim(T_ASK, T_idx, 2, axis = -1)
    coeffs = lax.dynamic_slice_in_dim(coeffs_all, T_idx, 2, axis = -1)
    a = coeffs[:47]
    c_sigma = coeffs[47:57]
    c_other = coeffs[57:]
    return (T_slice, a, c_other, c_sigma)

T_slice, a, c_other, c_sigma = slice_coeffs(0.5)
    
def f1(Mw, R_rup, a,
       M1):
    c4_mag = jnp.clip(c4 - (c4 - 1.0) * (5. - Mw), min = 1., max = c4)
    R = jnp.sqrt(R_rup ** 2 + c4_mag ** 2)
    base = a[1] + a[17] * R_rup
    # A coefficient
    f1A = jnp.where(Mw > M1, A5, A4)
    # Magnitude differences for low-magnitude case
    MaxM = jnp.where(Mw < M2, M2, Mw)
    MM1 = MaxM - M1
    Msq = (8.5 - MaxM) ** 2
    MwM2 = Mw - M2
    return base + f1A * MM1 + a[8] * Msq + jnp.where(Mw < M2, a[6], 0.) * MwM2 + (a[2] + A3 * MM1) * jnp.log(R)

def f7_8(Mw, SOF_flag,
         a):
    lnSA_SOF = a[12] * jnp.clip(Mw - 4, min = 0, max = 1)
    return jnp.where(SOF_flag == 1., lnSA_SOF, 0.)

def f5(vs30, SA_rock,
       a, c_other):
    _, b, c, _, v_lin, v1= c_other
    vs30_star = jnp.clip(vs30, max = v1)
    return a[10] * jnp.log(vs30_star / v_lin) - b * jnp.log(SA_rock + c) + b * jnp.log(SA_rock + c * (vs30_star / v_lin) ** N)

def f4(Mw, width, dip, R_jb, R_x, z_tor,
       a):
    T1 = jnp.clip(2 - (dip / 45), min = 4 / 3)

    T2 = 1 + A2_HW * (Mw - 6.5) + jnp.where(Mw < 6.5, (1 - A2_HW) * (Mw - 6.5) ** 2, 0.)
    
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

def _f_dAmp(vs30, SA_rock,
           c_other):
    _, b, c, _, v_lin, _= c_other
    dAmp = (-b * SA_rock) / (SA_rock + c) + (b * SA_rock) / (SA_rock + c * (vs30 / v_lin) ** N)
    return jnp.where(vs30 >= v_lin, 0., dAmp)

def _f_lnSA_SA_rock(Mw, width, dip, z_tor, SOF_flag,
                   vs30, z1p0,
                   R_jb, R_rup, R_x,
                   a, c_other
                   ):
    M1, b, _, c4, v_lin, v1= c_other
    c4_mag = jnp.clip(c4 - (c4 - 1) * (5 - Mw), min = 1, max = c4)
    vs_rock_capped = jnp.clip(vs_rock, max = v1)
    lnSA5_rock = (a[10] + b * N) * jnp.log(vs_rock_capped / v_lin)
    
    lnSA1 = f1(Mw, R_rup, a, M1)
    lnSA7_8 = f7_8(Mw, SOF_flag, a)
    lnSA4 = f4(Mw, width, dip, R_jb, R_x, z_tor, a)
    lnSA6 = f6(z_tor, a)
    lnSA10 = f10(vs30, z1p0, a)

    SA_rock = jnp.exp(lnSA1 + lnSA7_8 + lnSA5_rock + lnSA4 + lnSA6)

    lnSA5 = f5(vs30, SA_rock, a, c_other)

    lnSA = lnSA1 + lnSA7_8 + lnSA5 + lnSA4 + lnSA6 + lnSA10

    return lnSA, SA_rock

def _f_sigma(Mw, vs30, vs30inf_flag, SA_rock,
            sigma_coeffs, c_other):
    s, s_est = sigma_coeffs[:-3], sigma_coeffs[-3:]
    dAmp_p1 = _f_dAmp(vs30, SA_rock, c_other) + 1
    vs30_s = jnp.where(vs30inf_flag == 1., s_est, s[:3])

    phi_A = jnp.clip(vs30_s[1] + ((vs30_s[2] - vs30_s[1]) / 2) * (Mw - 4.0), min = s[1], max = s[2])
    phi_B_sq = phi_A ** 2 - phi_amp_sq
    phi_sq = phi_B_sq * dAmp_p1 ** 2 + phi_amp_sq

    tau_B = jnp.clip(s[3] + ((s[4] - s[3]) / 2) * (Mw - 4.0), min = s[3], max = s[4])
    tau = tau_B * dAmp_p1

    return (phi_sq + tau ** 2) ** (1 / 2)

def gmm_ASK14(Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
    T_slice, a, c_other, sigma_coeffs = slice_coeffs(T) 

    SOF_flag = fault.calc_SOF_flag()
    R_jb, R_rup, R_epi, R_hyp, R_x = R
    
    lnSA, SA_rock = _f_lnSA_SA_rock(Mw, fault.width, fault.dip, fault.z_tor, SOF_flag,
                                   site.vs30, site.z1p0, R_jb, R_rup, R_x,
                                   a, c_other)
    std = _f_sigma(Mw, site.vs30, site.vs30inf_flag, SA_rock,
                  sigma_coeffs, c_other)
    lnSA = jnp.interp(T, T_slice, lnSA)
    std = jnp.interp(T, T_slice, std)
    return lnSA, std