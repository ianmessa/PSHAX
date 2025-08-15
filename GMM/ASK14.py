import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
import re

from GMM.gmm_scenario import *

##### GROUND MOTION COEFFICIENTS #####
gmc = pl.read_csv('ask14_coeffs.csv')
gmc_col = gmc.columns
gmc = gmc.to_jax()

# First few
T, V_lin, b, n, c = gmc[:, :5].T
c4 = gmc[:, 5]
M = gmc[:, 6:8].T

# Lots of a_ns, so we'll make an array for them.
#   Start with the actual columns.
a_sparse = gmc[:, 8:38].T
# Strip indices from column titles. (One is a capital letter but we're making
#   it lowercase).
a_idcs = jnp.array([int(re.findall('a' + r'\d+', a_col.lower())[0][1:]) for a_col in gmc_col[8:38]])
# Make empty array
a_empty = jnp.zeros((47, a_sparse.shape[-1]))
# Save values.
# INDICES MATCH COEFFICIENT NAMES IN THE PAPER. FIRST ROW WILL BE EMPTY.
a = a_empty.at[a_idcs].set(a_sparse)

# Same thing for s_n, but it's easier.
# AGAIN, INDICES MATCH COEFFICIENT NAMES IN THE PAPER. FIRST ROW WILL BE EMPTY.
s_est = gmc[:, 38:40].T
s_empty = jnp.zeros((1, s_est.shape[-1]))
s_est = jnp.insert(s_est, 0, s_empty, axis = 0)
s = gmc[:, 40:].T
s = jnp.insert(s, 0, s_empty, axis = 0)

# Also defining V_1 so we don't need to do it twice
v_1 = jnp.exp(-0.35 * jnp.log(T / 0.5) + jnp.log(1500))
v_1 = v_1.at[T <= 0.5].set(1500)
v_1 = v_1.at[T >= 3].set(800)

coeffs = [T, V_lin, b, n, c, c4, M, a, s_est, s, v_1]

##### MAIN MODEL #####
# Base model (Section 4.1)
def fn_1(c4, M, a, 
         mag, R_rup):
    # Get c4
    c4_which = jnp.array([mag >= 5, 
                          mag >= 4, 
                          mag >=0])
    # (Make condition exclusive. This trick will be used throughout.)
    c4_idx = jnp.argmax(c4_which)
    c4_returns = jnp.array([c4, 
                            c4 - (c4-1) * (5 - mag), 
                            jnp.ones_like(c4)])
    c4_mag = lax.dynamic_slice(c4_returns, (c4_idx, 0), (1, c4.shape[0]))[0]

    # Get R
    R = jnp.sqrt(R_rup ** 2 + c4_mag **2)

    # Get base
    base_which = jnp.array([jnp.all(mag <= M[2]),
                            jnp.all(mag <= M[1]), 
                            jnp.all(M > M[2])])
    # Render exclusive since M1 
    base_which = jnp.argmax(base_which)

    def base_return_1(mag, R_rup): 
        return (a[1] + 
                a[6] * (mag - M[2]) + 
                a[7] * (mag - M[2]) ** 2 + 
                a[4] * (M[2] - M[1]) + 
                a[8] * (8.5 - M[2]) **2 + 
                (a[2] + a[3] * (M[2] - M[1])) * jnp.log(R) + 
                a[17] * R_rup)
    def base_return_2(mag, R_rup):
        return (a[1] + 
                a[4] * (mag - M[1]) + 
                a[8] * (8.5 - mag) ** 2 + 
                (a[2] + a[3] * (M[2] - M[1])) * jnp.log(R) + 
                a[17] * R_rup)
    def base_return_3(mag, R_rup):
        return (base_return_2(mag, R_rup) - 
                a[4] * (mag - M[1]) + 
                a[5] * (mag - M[1]))
    
    base_returns = ([base_return_1, 
                     base_return_2, 
                     base_return_3])
    
    base = lax.switch(base_which, base_returns, mag, R_rup)

    return base

# Hanging wall model
#   May need to refactor using housed functions.
def fn_4(a,
         HW_flag, dip, W, mag, Z_TOR, R_x, R_jb, R_y0):
    # If HW flag:
    def HW_true(dip, W, mag, Z_TOR, R_x, R_jb, R_y0):
        # Get hanging wall taper 1 (4.11)
        HW_tpr1 = jnp.max(jnp.array([60, 90 - dip])) / 45
        
        # Get hanging wall taper 2 (4.12)
        HW_a2 = 0.2
        HW_tpr2_which = jnp.array([mag > 6.5, 
                                    mag > 5.5, 
                                    mag > 0])
        HW_tpr2_idx = jnp.argmax(HW_tpr2_which)
        HW_tpr2_base = 1 + HW_a2 * (mag - 6.5)
        HW_tpr2_returns = jnp.array([HW_tpr2_base,
                                     HW_tpr2_base - (1 - HW_a2) * (mag - 6.5) ** 2, 
                                     0])
        HW_tpr2 = lax.dynamic_slice(HW_tpr2_returns, (HW_tpr2_idx,), (1,))[0]
        
        # Get hanging wall taper 3 (4.13)
        h1, h2, h3 = 0.25, 1.5, -0.75
        R1 = W * jnp.cos(dip * jnp.pi)
        R2 = 3 * R1
        HW_tpr3_which = jnp.array([R_x <= R1,
                                R_x <= R2,
                                R_x > R2])
        HW_tpr3_idx = jnp.argmax(HW_tpr3_which)
        HW_tpr3_returns = jnp.array([h1 + h2 * (R_x / R1) + h3 * (R_x / R1) ** 2,
                                     1 - (R_x - R1)/(2 * R1),
                                     0])
        HW_tpr3 = lax.dynamic_slice(HW_tpr3_returns, (HW_tpr3_idx,), (1,))[0]
        
        # And taper 4 (eq 4.14)...
        HW_tpr4_Z_term = jnp.clip((Z_TOR ** 2) / 100, min = 0, max = 1)
        HW_tpr4 = 1 - HW_tpr4_Z_term

        # And taper 5 thank god (eq 4.15)
        # Oh it's two parts. Here's the first (4.15a)...
        R_y1 = R_x * jnp.tan(20 * jnp.pi / 180)
        HW_tpr5a_R_term = jnp.clip((R_y0 - R_y1) / 5, min = 0, max = 1)
        HW_tpr5a = 1 - HW_tpr5a_R_term

        # And the second (4.15b).
        HW_tpr5b_R_jb = jnp.clip(R_jb / 30, min = 0, 
                            max = 1)
        HW_tpr5b = 1 - HW_tpr5b_R_jb

        # Conditions.
        HW_tpr5 = lax.select(R_y0 >= 0, HW_tpr5a, HW_tpr5b)

        return a[13] * HW_tpr1 * HW_tpr2 * HW_tpr3 * HW_tpr4 * HW_tpr5
    
    # If not:
    def HW_false(dip, W, mag, Z_TOR, R_x, R_jb, R_y0):
        return a[13] * 0
    
    return lax.cond(HW_flag, HW_true, HW_false, dip, 
                                           W, 
                                           mag, 
                                           Z_TOR, 
                                           R_x, R_jb, R_y0)

# Site response
def fn_5(V_lin, b, n, c, a, v_1,
         vs30, SA):
    V_star= jnp.clip(v_1, min = 0, max = vs30)

    site_1 = (a[10] + b * n) * jnp.log(V_star/ V_lin)
    site_2 = (a[10] * jnp.log(V_star/ V_lin) - 
                      b * jnp.log(SA + c) + 
                      b * jnp.log(SA + c * (V_star/ V_lin) ** n))
    
    site = lax.select(V_lin >= vs30, site_2, site_1)

    return site

# Z_TOR model
def fn_6(a, 
         Z_TOR):
    return a[15] * jnp.clip(Z_TOR / 20, min = 0, max = 1)

# Style of Faulting model
def fn_7_fn_8(a,
              mag, SOF):
    # Combine normal and reverse.
    #   Matlab is verbose!
    F_SOF = jnp.round(jnp.abs(SOF) + 1e-12)
    a_SOF = lax.select(SOF > 0, a[11], a[12])
    mag_SOF = jnp.clip(mag - 4, min = 0, max = 1)

    return F_SOF * a_SOF * mag_SOF

# Soil depth model
def fn_10(
          z1, vs30, region):
    const = lax.select(region == 10, 
                                           jnp.array([-5.23, 2., 412.]),
                                           jnp.array([-7.67, 4., 610.]))
    
    z1_ref = jnp.exp(const[0] / const[1] * 
                     jnp.log((vs30 ** const[1] + const[2] ** const[1]) / 
                             1360 ** const[1] + const[2] ** const[1]))
    z1_ref = z1_ref / 1000
    
    return lax.select(z1 < 0, z1_ref, z1)

# Regional model
def fn_regional(V_lin, a, v_1,
                region, vs30, R_rup):
    fn_12, fn_13 = 0, 0
    def twn(vs30):
        V_star= jnp.clip(v_1, min = 0, max = vs30)
        fn_12 = a[31] * jnp.log(V_star/ V_lin)
        return fn_12 * a[25] * R_rup
    
    def cn(vs30):
        return a[28] * R_rup
    
    def jpn(vs30):
        vs_mid_bins = jnp.array([150, 250, 350, 450, 600, 850, 1150, 2000])
        delta = jnp.concatenate([a[36:43],a[43][None]])
        fn_13 = jnp.stack([jnp.interp(vs30, vs_mid_bins, delta.T[i]) for i in range(delta.shape[1])])
        return fn_13 * a[29] * R_rup
    
    def other(vs30):
        return jnp.zeros_like(a[25], dtype = float)
    
    # Branches for all regions
    region_deltas = [other] * 13
    region_deltas[3] = twn
    region_deltas[9] = cn
    region_deltas[10] = jpn

    # We're done.
    return lax.switch(region, region_deltas, vs30)

# lnSA model
def fn_ASK14_lnSA(V_lin, b, n, c, c4, M, a, v_1,
          mag, W, dip, Z_TOR, SOF, HW_flag,
          R_rup, R_jb, R_x, R_y0, 
          vs30, SA, z1, region:str):
    fn_1_out = fn_1(c4, M, a, 
                    mag, R_rup)
    fn_4_out = fn_4(a, 
                    HW_flag, dip, W, mag, Z_TOR, R_x, R_jb, R_y0)
    fn_5_out = fn_5(V_lin, b, n, c, a, v_1, 
                    vs30, SA)
    fn_6_out = fn_6(a, 
                    Z_TOR)
    fn_7_fn_8_out = fn_7_fn_8(a, 
                              mag, SOF)
    fn_10_out = fn_10(
                      z1, vs30, region)
    regional_out = fn_regional(V_lin, a, v_1,
                               region, vs30, R_rup)
    return sum([fn_1_out, fn_4_out, fn_5_out, fn_6_out, fn_7_fn_8_out, fn_10_out, regional_out])

#### ALEATORIC UNCERTAINTY ####
def fn_ASK14_sig(V_lin, b, n, c, s, s_est, 
           region:int, mag:float, R_rup:float, vs30:float, vs30_flag:bool, 
           SA1180:jnp.ndarray):
    s_fSig = lax.select(vs30_flag, s_est, s[:3])

    def phi_AL_all(magmag:float, R_rupmag:float):
        mag_phi_AL = jnp.clip(mag, min = 4, max = 6)
        phi_AL = lax.select(mag_phi_AL <= 6, s_fSig[1] + (s_fSig[2] - s_fSig[1]) / (2 * (mag - 4)), s_fSig[2])
        return phi_AL
    
    def phi_AL_jp(mag:float, R_rup:float):
        phi_AL_which = jnp.array([R_rup < 30,
                                  R_rup <= 80,
                                  R_rup > 80])
        phi_AL_idx = jnp.argmax(phi_AL_which)
        phi_AL_returns = jnp.array([s[5],
                                    s[5] + (s[6] - s[5]) / 50 * (R_rup - 30),
                                    s[6]])
        phi_AL = lax.dynamic_slice(phi_AL_returns, (phi_AL_idx, 0), (1, s.shape[-1]))[0]
        return phi_AL

    phi_AL = lax.cond(region == 10, phi_AL_all, phi_AL_jp, mag, R_rup)

    tau_B_which = jnp.array([mag < 5,
                             mag <= 7,
                             mag > 7])
    tau_B_idx = jnp.argmax(tau_B_which)
    tau_B_returns = jnp.array([s[3],
                               s[3] + (s[4] - s[3]) / 2 * (mag - 5), 
                               s[4]])
    tau_B = lax.dynamic_slice(tau_B_returns, (tau_B_idx, 0), (1, s.shape[-1]))[0]

    phi_B = (phi_AL ** 2 - 0.4 ** 2) ** (1/2)
    drv_amp_SA1180 = b * SA1180 * (-1 / (SA1180 + c) + 1/(SA1180 + c * (vs30 / V_lin) ** n))
    drv_bool = V_lin > vs30
    drv_zeros = jnp.zeros_like(drv_amp_SA1180)
    drv_amp_SA1180 = lax.select(drv_bool, drv_amp_SA1180, drv_zeros)

    phi = (phi_B ** 2 * (1 + drv_amp_SA1180) ** 2 + 0.4 ** 2) ** (1/2)
    tau = tau_B * (1 + drv_amp_SA1180)
    sig = (phi **2 + tau **2) ** (1/2)
    return sig

#### FINAL MODEL ####
def fn_ASK14(scenario:gmm_scenario, coeffs:list = coeffs):

    T, V_lin, b, n, c, c4, M, a, s_est, s, v_1 = coeffs

    SA1180 = jnp.exp(fn_ASK14_lnSA(V_lin, b, n, c, c4, M, a, v_1,
                              mag, W, dip, Z_TOR, SOF, HW_flag,
                              R_rup, R_jb, R_x, R_y0, 
                              1180., 0., -1., region))
    
    lnSA = fn_ASK14_lnSA(V_lin, b, n, c, c4, M, a, v_1,
                     Mw, W, dip, Z_TOR, SOF, HW_flag,
                     R_rup, R_jb, R_x, R_y0, 
                     vs30, SA1180, z1, region)
    
    std = fn_ASK14_sig(V_lin, b, n, c, s, s_est, 
                 region, mag, R_rup, vs30, vs30_flag, SA1180)

    return T, lnSA, std