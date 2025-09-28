import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jrnd

import polars as pl 

from gm_utils import gm_scenario

gmc = pl.read_csv('CY14_coeffs.csv')
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T
T = gmc[0]
empty = jnp.zeros_like(T)

c_RB = gmc[4]
c_n, c_M, c_HM = gmc[jnp.array([12, 13, 16])]
c = jnp.empty((12, 5, T.shape[0]))
# c1, c1a - d
c = c.at[1].set(gmc[7:12])
# c2
c = c.at[2, 0].set(gmc[1])
# c3
c = c.at[3, 0].set(gmc[14])
# c4
c = c.at[4, 0:2].set(gmc[2:4])
# c5 - c7
c = c.at[5:8, 0].set(gmc[jnp.array([15, 17, 18])])
# c8
c = c.at[8, 0:3].set(gmc[jnp.array([5, 6, 20])])
# c9, c9a, c9b
c = c.at[9, 0:3].set(gmc[21:24])
# c11, c11b
c = c.at[11, jnp.array([0, 2])].set(gmc[24:26])
# c_gamma
c_gamma = jnp.insert(gmc[26:29], 0, empty, axis = 0)
phi = jnp.insert(gmc[29:35], 0, empty, axis = 0)
tau = jnp.insert(gmc[35:37], 0, empty, axis = 0)
sigma = jnp.insert(gmc[37:40], 0, empty, axis = 0)
sigma2_JP, gamma_JP_IT, gamma_WN = gmc[40:43]
phi1_JP, phi5_JP, phi6_JP = gmc[43:]

z_tor_const_RV = jnp.array([2.704, 1.226, 5.849])
z_tor_const_NM = jnp.array([2.673, 1.136, 4.97])

A = 571 ** 4
B = 1360 ** 4 + A

def f_SA_ref(Mw, dip, R_jb, R_rup, R_x, z_tor, SOF_flag):
    r1 = c[1, 0] + c[2, 0] * (Mw - 6) + ((c[2, 0] - c[3, 0]) / c_n) * \
        jnp.log(1 + jnp.exp(c_n * (c_M - Mw)))
    r2 = c[4, 0] * jnp.log(R_rup + c[5, 0] * jnp.cosh(c[6, 0] * jnp.clip(Mw - c_HM, min = 0)))

    gamma = c_gamma[1] + c_gamma[2] / jnp.cosh(jnp.clip(Mw - c_gamma[3], min = 0))
    r3 = (c[4, 1] - c[4, 0]) * jnp.log((R_rup ** 2 + c_RB ** 2) ** (1 / 2)) + R_rup * gamma

    cosh_Mw = jnp.cosh(2 * jnp.clip(Mw - 4.5, min = 0))
    cos_dip = jnp.cos(jnp.deg2rad(dip))
    z_tor_const = lax.select(SOF_flag == -1, z_tor_const_RV, z_tor_const_NM)
    z_tor_Mw = jnp.clip(z_tor_const[0] - z_tor_const[1] * (Mw - z_tor_const[2]), min = 0, max = z_tor_const[0]) ** 2
    dz_tor = z_tor - z_tor_Mw
    c_SOF = lax.select(SOF_flag == -1, c[1, 1:4:2], c[1, 2:5:2]) * jnp.abs(SOF_flag)

    r4 = dz_tor * (c[7, 0] + c[7, 2] / cosh_Mw) + cos_dip ** 2 * (c[11, 0] + c[11, 2] / cosh_Mw) + c_SOF[0] + c_SOF[1] / cosh_Mw
    r5 = c[9, 0] + cos_dip * \
        (c[9, 1] + (1 - c[9, 1]) * jnp.tanh(R_x / c[9, 2])) * \
        (1 - (R_jb ** 2 + z_tor ** 2) ** (1 / 2) / (R_rup + 1))
    
    return jnp.exp(r1 + r2 + r3 + r4 + r5)

def f_soil_nl(vs30):
    exp1 = jnp.exp(phi[3] * (jnp.clip(vs30, max = 1130) - 360))
    exp2 = jnp.exp(phi[3] * (1130 - 360))
    return phi[2] * (exp1 - exp2)

def f_lnSA(scn:gm_scenario):
    soil_lin = phi[1] * jnp.clip(jnp.log(scn.vs30 / 1130), max = 0)
    soil_nl = f_soil_nl(scn.vs30)
    SA_ref = f_SA_ref(scn.Mw, scn.dip,
                      scn.R_jb, scn.R_rup, scn.R_x,
                      scn.z_tor, scn.SOF_flag)
    soil_nl_mod = soil_nl * jnp.log((SA_ref + phi[4]) / phi[4])
    z1_ref = jnp.exp(-7.15 / 4 * jnp.log((scn.vs30 ** 4 + A) / B))
    dz1 = (scn.z1p0 * 1000) - z1_ref
    rk_depth = phi[5] * (1 - jnp.exp(- dz1 / phi[6]))

    return jnp.log(SA_ref) + soil_lin + soil_nl_mod + rk_depth

def f_sigma(scn:gm_scenario):
    return 0

def f_CY14(scn:gm_scenario):
    return T, f_lnSA(scn), f_sigma(scn)