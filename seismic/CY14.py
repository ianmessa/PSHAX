import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jrnd
from jax.typing import ArrayLike

import polars as pl 
from importlib.resources import files

from seismic.gm_utils import *

gmc = pl.read_csv(files("seismic") / "CY14_coeffs.csv")
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc = gmc.sort('T')
gmc_col = gmc.columns
gmc_df = gmc
gmc = gmc.cast(pl.Float64).to_jax().T
T_CY = gmc[0]
empty = jnp.zeros_like(T_CY)

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
# c7b
c = c.at[7, 2].set(gmc[19])
# c8
c = c.at[8, 0:3].set(gmc[jnp.array([5, 6, 20])])
# c9, c9a, c9b
c = c.at[9, 0:3].set(gmc[21:24])
# c11, c11b
c = c.at[11, jnp.array([0, 2])].set(gmc[24:26])
# c_gamma
c_gamma = jnp.insert(gmc[26:29], 0, empty, axis = 0)
c_phi = jnp.insert(gmc[29:35], 0, empty, axis = 0)
c_tau = jnp.insert(gmc[35:37], 0, empty, axis = 0)
c_sigma = jnp.insert(gmc[37:40], 0, empty, axis = 0)
c_sigma2_JP, c_gamma_JP_IT, c_gamma_WN = gmc[40:43]
c_phi1_JP, c_phi5_JP, c_phi6_JP = gmc[43:]

z_tor_const_RV = jnp.array([2.704, 1.226, 5.849])
z_tor_const_NM = jnp.array([2.673, 1.136, 4.97])

A = 571. ** 4
B = 1360. ** 4 + A

def f_SA_ref(Mw, dip, z_tor, SOF_flag,  R_jb, R_rup, R_x, ):
    dip_rad = jnp.deg2rad(dip)

    r1 = c[1, 0] + c[2, 0] * (Mw - 6.) + ((c[2, 0] - c[3, 0]) / c_n) * jnp.log(1 + jnp.exp(c_n * (c_M - Mw)))
    r2 = c[4, 0] * jnp.log(R_rup + c[5, 0] * jnp.cosh(c[6, 0] * jnp.maximum(Mw - c_HM, 0.)))
    gamma = c_gamma[1] + c_gamma[2] / jnp.cosh(jnp.maximum(Mw - c_gamma[3], 0.))
    r3 = (c[4, 1] - c[4, 0]) * jnp.log(jnp.sqrt(R_rup ** 2 + c_RB ** 2)) + R_rup * gamma

    cosh_Mw = jnp.cosh(2 * jnp.maximum(Mw - c_gamma[3], 0.))
    cos_dip = jnp.cos(dip_rad)

    # Calculate Mw_z_tor (separate fn in nshmp-haz)
    is_rev = SOF_flag == -1.
    Mw_z_tor_rev = lambda Mw: jnp.clip(2.704 - 1.226 *(Mw - 5.849) , min = 0., max = 2.704)
    Mw_z_tor_else = lambda Mw: jnp.clip(2.673 - 1.136 *(Mw - 4.970) , min = 0., max = 2.673)
    Mw_z_tor = lax.cond(is_rev, Mw_z_tor_rev, Mw_z_tor_else, Mw)

    dz_tor = z_tor - Mw_z_tor

    r4 = ((c[7, 0] + c[7, 2]) * dz_tor + (c[11, 0] + c[11, 2]) * cos_dip ** 2) / cosh_Mw
    r5_cond = R_x >= 0.
    r5 = c[9, 0] * jnp.cos(dip_rad) * \
        (c[9, 1] + (1 - c[9, 1]) * jnp.tanh(R_x / c[9, 2])) * \
        (1 - jnp.sqrt(R_jb ** 2 + z_tor ** 2) / (R_rup + 1.))
    r5 = jnp.where(r5_cond, r5, 0.)
    
    return jnp.exp(r1 + r2 + r3 + r4 + r5)

def f_lnSA(vs30, z1p0, SA_ref, soil_nonlin):
    soil_lin = c_phi[1] * jnp.minimum(jnp.log(vs30 / 1130.), 0.)
    soil_nonlin_mod = soil_nonlin * jnp.log((SA_ref + c_phi[4]) / c_phi[4])

    # Calculate delta z1p0 (separate fn in nshmp-haz)
    z1p0_ref = jnp.exp(-7.15 / 4 * jnp.log((vs30 ** 4 + A) / B))
    dz1p0 = (z1p0 * 1000) - z1p0_ref

    rk_depth = c_phi[5] * (1. - jnp.exp(-dz1p0 / c_phi[6]))

    return jnp.log(SA_ref) + soil_lin + soil_nonlin_mod + rk_depth

def f_std(Mw, vs30inf_flag, SA_ref, soil_nonlin):
    nonlin_0 = soil_nonlin * SA_ref / (SA_ref + c_phi[4])
    nonlin_0sq = (1 + nonlin_0) ** 2
    Mw_thresh = jnp.clip(Mw - 5., min = 0, max = 1.5)
    tau = c_tau[1] + (c_tau[2] - c_tau[1]) / 1.5 * Mw_thresh

    sig_nonlin_0 = c_sigma[1] + (c_sigma[2] - c_sigma[1]) / 1.5 * Mw_thresh
    vs_term = jnp.where(vs30inf_flag == 1., c_sigma[3], 0.7)
    sig_nonlin_0 = sig_nonlin_0 * jnp.sqrt(vs_term + nonlin_0sq)

    return (tau ** 2 * nonlin_0sq + sig_nonlin_0 ** 2)

def f_CY14(Mw:float, site:Site, fault:Fault, R:jax.Array):
    R_jb, R_rup, R_epi, R_hyp, R_x = R
    SA_ref = f_SA_ref(Mw, fault.dip, fault.z_tor, fault.calc_SOF_flag(), R_jb, R_rup, R_x)

    # Calculate nonlinear attenuation effect (separate fn in nshmp-haz)
    soil_nonlin1 = jnp.exp(c_phi[3] * (jnp.minimum(site.vs30, 1130.) - 360.))
    soil_nonlin2 = jnp.exp(c_phi[3] * (1130. - 360.))
    soil_nonlin = c_phi[2] * (soil_nonlin1 - soil_nonlin2)

    lnSA = f_lnSA(site.vs30, site.z1p0, SA_ref, soil_nonlin)
    std = f_std(Mw, site.vs30inf_flag, SA_ref, soil_nonlin)

    lnSA = jnp.interp(T_master, T[T_sort], lnSA[T_sort])
    std = jnp.interp(T_master, T[T_sort], std[T_sort])

    return lnSA, std
