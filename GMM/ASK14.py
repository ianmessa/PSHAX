import jax
from jax import lax
from jax.tree_util import Partial
from jax import numpy as jnp

import polars as pl

from gm_scenario import *

from matplotlib import pyplot as plt

##### GROUND MOTION COEFFICIENTS #####
gmc = pl.read_csv('ASK14_coeffs.csv')
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

# First few
T, v_lin, b, n, M1, c, c4 = gmc[:7]
M = jnp.empty((3, T.shape[0]))
M = M.at[1].set(M1)
M = M.at[2].set(5.0)
# Update M with constant for M2 and 0-row for M0
empty = jnp.zeros_like(T)

# Get a. But they're all out of order...
a = gmc[7:38]
a_idcs = jnp.array([int(ai[1:]) for ai in gmc_col[7:38]])
a = a[jnp.argsort(a_idcs)]
a_missing = jnp.arange(a_idcs.max())
a_missing = a_missing[~jnp.isin(a_missing, a_idcs)]
# Update indices to account for insertions
a_missing = a_missing - jnp.arange(a_missing.shape[0])
a = jnp.insert(a, a_missing, empty, axis = 0)

# Grab s
s_est = gmc[38:40]
s_est = jnp.insert(s_est, 0, empty, axis = 0)
s = gmc[40:]
s = jnp.insert(s, 0, empty, axis = 0)

# Also defining v1 so we don't need to do it twice
v1 = jnp.exp(-0.35 * jnp.log(T / 0.5) + jnp.log(1500))
v1 = v1.at[T <= 0.5].set(1500)
v1 = v1.at[T >= 3].set(800)

# Works with for loop
# Doesn't work with vector index

#### MAIN MODEL ####
# Base model (section 4.1)
def f1(Mw, R_rup):
    # Just clip c4_magnitude instead of if/else
    c4_mag = jnp.clip(c4 - (c4 - 1) * (5 - Mw), min = c4, max = 1)
    R = (R_rup ** 2 + c4_mag ** 2) ** (1/2)

    # Common among all cases
    base = a[1] + (a[2] + a[3] * (Mw - M[1])) * jnp.log(R) + a[17] * R_rup

    # Double leq conditional index
    cond = (Mw <= M[2]).astype(int) + (Mw <= M[1]).astype(int)
    
    # Conditional elements
    a_MM = lax.select_n(cond, a[5] * (Mw - M[1]), a[4] * (Mw - M[1]), a[6] * (Mw - M[2]))
    a_MM2 = lax.select(Mw <= M[2], a[7] * (8.5 - M[2]) ** 2, a[8] * (8.5 - Mw) ** 2)
    a_MM_min = lax.select(Mw <= M[2], a[4] * (M[2] - M[1]) + a[8] * (8.5 - M[2]) ** 2, empty)
    
    # Sum base and conditionals to get final gmp
    return base + a_MM + a_MM2 + a_MM_min

# SOF (section 4.2)
#   Combining f7, f8 into one fn
def f7_8(Mw, SOF):
    # Just double-select a instead of two separate functions,
    #   many of whose branches return 0s
    a_SOF = lax.select(SOF < 0, a[11], a[12])
    a_SOF = lax.select(jnp.abs(SOF) > 0.5, a_SOF, empty) 
    # Just clip instead of if/elseif/else
    return a_SOF * jnp.clip(Mw - 4, min = 0, max = 1)

# Site response (4.3)
def f5(SA_rock, vs30):
    # Straightforward
    vs30_star = jnp.clip(vs30, max = v1)

    site = (a[10] + b * n) * jnp.log(vs30_star / v_lin)
    site2 = site - \
            b * jnp.log(SA_rock + c) + \
            b * jnp.log(SA_rock + c * (vs30_star / v_lin) ** n)
    
    return site#lax.select(vs30 < v_lin, site, site2)

# Hanging wall (4.4)
    # Only calculate if HW_flag is 1
def f4(Mw, dip, width, 
       R_x, R_jb, R_y0, 
       z_tor):
    # Clip insetad of if/else
    HW_taper1 = jnp.clip((90 - dip) / 45, max = 60 / 45)

    # Once again, clip instead of if/else
    HW_taper2_max = 1 + 0.2 * (Mw - 6.5)
    HW_taper2_main = HW_taper2_max - 0.8 * (Mw - 6.5) ** 2
    HW_taper2 = jnp.clip(HW_taper2_main, min = 0, max = HW_taper2_max)

    # Ugh...
    h1, h2, h3 = 0.25, 1.5, -0.75
    R1 = width * jnp.cos(dip * jnp.pi / 180)
    R2 = 3 * R1
    HW_taper3_main = 1 - (R_x / R1) / (R2 - R1)
    HW_taper3_min = h1 + h2 * (R_x / R1) + h3 * (R_x / R1) ** 2
    cond = (R_x < R2).astype(int) + (R_x <= R1).astype(int)
    HW_taper3 = lax.select_n(cond, 0., HW_taper3_main, HW_taper3_min)

    HW_taper4 = jnp.clip(1 - (z_tor ** 2) / 100, min = 0)

    # Last one!...
    R_y1 = R_x * jnp.tan(20 * jnp.pi / 180)
    HW_taper5_R_y0 = jnp.clip(1 - (R_y0 - R_y1) / 5, min = 0, max = 1)
    HW_taper5_else = jnp.clip(1 - R_jb / 30, min = 0, max = 1)
    HW_taper5 = lax.select(R_y0 >= 0, HW_taper5_R_y0, HW_taper5_else)

    return a[13] * HW_taper1 * HW_taper2 * HW_taper3 * HW_taper4 * HW_taper5

# Top of rupture model (section 4.5)
def f6(z_tor):
    return a[15] * jnp.clip(z_tor / 20, max = 1)

# Soil depth model (section 4.6; only changes anything if z1p0 < 0)
def f10(vs30, 
        z1p0, 
        region):
    # Also straightforward. See matlab implementation.
    z1p0_ref_10 = jnp.exp(-5.23 / 2 * jnp.log((vs30 ** 2 + 412 ** 2) / (1360 ** 2 + 412 ** 2))) / 1000
    z1p0_ref_else = jnp.exp(-7.67 / 4 * jnp.log((vs30 ** 2 + 610 ** 2) / (1360 ** 2 + 610 ** 2))) / 1000
    z1p0_ref = lax.select(region == 10, z1p0_ref_10, z1p0_ref_else)
    ret = lax.select(z1p0 < 0., z1p0_ref, z1p0)
    return jnp.full_like(T, ret)

# Regional model (4.8)
def regional(R_rup, 
             vs30, 
             region):
    # Big one. Convert regional indices from (all else, 3, 9, 10) to (0, 1, 2, 3)
    region_post = jnp.sum(jnp.cumsum(jnp.array([region == i for i in [3, 9, 10]])))

    # Taiwan:
    def f_TW():
        # vs30* scaling
        vs30_star = jnp.clip(vs30, min = v1)
        f12 = a[31] * jnp.log(vs30_star / v_lin)
        delta_TW = f12 + a[25] * R_rup
        return delta_TW

    def f_CN():
        # easy one
        delta_CN = a[28] * R_rup
        return delta_CN

    def f_JP():
        # Identify midpoints of vs30 bins
        JP_vs_mid = jnp.array([150, 250, 350, 450, 600, 850, 1150, 2000])
        # Slam coefficients together
        a_JP = jnp.concatenate([a[36:43], a[42:43]]).T
        # Interpolate coefficients based on bins
        f13 = jax.vmap(Partial(jnp.interp, x = vs30, xp = JP_vs_mid))(fp = a_JP)
        return f13 + a[29] * R_rup
    
    # Other regions, no modifier
    f_empty = lambda: empty

    # Select region
    delta = lax.switch(region_post, [f_empty, f_TW, f_CN, f_JP])
    
    return delta

def f_SA1180(Mw, width, dip, 
             R_jb, R_rup, R_x, R_y0,
             z_tor,
             SOF):
    lnSA1180 = f1(Mw, R_rup) + \
               f7_8(Mw, SOF) + \
               (a[10] + b * n) * jnp.log(1180 / v_lin) + \
               f4(Mw, dip, width,
                  R_x, R_jb, R_y0, 
                  z_tor) + \
               f6(z_tor)
    
    return jnp.exp(lnSA1180)

# Cumulative lnSA model
def f_lnSA(Mw, width, dip, 
        R_jb, R_rup, R_x, R_y0, 
        vs30, 
        z1p0, z_tor, SA1180,
        SOF, HW_flag, region):
    # Zeros for if HW_flag is false (0)
    f_filler = lambda *args: empty

    return (
           f1(Mw, R_rup) + 
           f7_8(Mw, SOF) + 
           f5(SA1180, vs30) + 
           lax.cond(HW_flag, f4, f_filler, Mw, dip, width, R_x, R_jb, R_y0, z_tor) + 
           f6(z_tor) +
           f10(vs30, z1p0, region) + 
           regional(R_rup, vs30, region)
        )

# StD model (7.1, 7.2)
def f_sigma(Mw, 
         R_rup, 
         vs30, vs30_flag, 
         region, SA1180):
    
    s1 = lax.select(vs30_flag, s[1], s_est[1])
    s2 = lax.select(vs30_flag, s[2], s_est[2])

    # Normal phi
    def phi_al_else():
        phi_al = s1 + (s2 - s1) / 2 * (Mw - 4)
        phi_al = jnp.clip(phi_al, min = s1, max = s2)
        return phi_al
    
    # Japan phi (just different tabular coeffs)
    def phi_al_JP():
        phi_al = s[5] + (s[6] - s[5]) / 50 * (R_rup - 30)
        phi_al = jnp.clip(phi_al, min = s[5], max = s[6])
        return phi_al
    
    # Select the right one
    phi_al = lax.select(region == 10, phi_al_JP(), phi_al_else())

    # Within-event StD
    phi_amp = 0.4
    phiB = (phi_al ** 2 - phi_amp ** 2) ** (1 / 2)

    dAmp_dSA1180 = b * SA1180 * (-1 / (SA1180 + c) + 1 / (SA1180 + c * (vs30 / v_lin) ** n))
    dAmp_dSA1180 = dAmp_dSA1180.at[v_lin <= vs30].set(0)

    phi = (phiB ** 2 * (1 + dAmp_dSA1180) ** 2 + phi_amp ** 2) ** (1 / 2)

    # Between-event StD
    tauB = s[3] + (s[4] - s[3]) / 2 * (Mw - 5)
    tauB = jnp.clip(tauB, min = s[3], max = s[4])
    tau = tauB * (1 + dAmp_dSA1180)

    # Cumulative StD
    sig = (phi ** 2 + tau ** 2) ** (1 / 2)
    return sig

# FULL MODEL
def ASK14(scn: gm_scenario):
    vs30_star = jnp.clip(scn.vs30, min = v1)

    # Calculate SA1180 ground motions
    SA1180 = f_SA1180(scn.Mw, scn.width, scn.dip,
                      scn.R_jb, scn.R_rup, scn.R_x, scn.R_y0,
                      scn.z_tor, scn.SOF)

    # Plug back into lnSA and std
    lnSA = f_lnSA(scn.Mw, scn.width, scn.dip, 
                  scn.R_jb, scn.R_rup, scn.R_x, scn.R_y0, 
                  scn.vs30, 
                  scn.z1p0, scn.z_tor, 
                  SA1180, scn.SOF, scn.HW_flag, scn.region)
    
    sigma = f_sigma(scn.Mw, scn.R_rup, 
                    scn.vs30, scn.vs30_flag, 
                    scn.region, SA1180)

    # We made it.
    return T, lnSA, sigma