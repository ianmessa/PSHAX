import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .seismic_utils import *

gmc = pl.read_csv(files("seismic") / "CB14_coeffs.csv")
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.

gmc = gmc.with_columns([pl.col("T").cast(pl.Float64)])
gmc = gmc.sort('T')
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T_CB = gmc[0]
empty_all = jnp.zeros_like(T_CB, dtype = float)
c_all = gmc[1:22]
# Region 0 (all others)/1/2/3
Dc20_CA_all, Dc20_JP_all, Dc20_CH_all = gmc[22:25]
a2_all = gmc[25]
h_all = gmc[26:32]
h_all = jnp.insert(h_all, 0, empty_all, axis = 0)
k_all = gmc[32:35]
k_all = jnp.insert(k_all, 0, empty_all, axis = 0)
C_all, n_all = gmc[35:37]
phi1_all, phi2_all = gmc[37:39]
tau1_all, tau2_all = gmc[39:41]
phi_lnAF_all, phi_C_all, rho_all = gmc[41:]

site_all = jnp.stack([C_all, n_all], axis = 0)
sigma_all = jnp.stack([tau1_all, tau2_all, 
                       phi1_all, phi2_all, 
                       phi_lnAF_all, rho_all], axis = 0)
attn_all = Dc20_CA_all

# Don't know why this is here...
cy_csim = 0.1

def slice_coeffs(T):
    T_idx = jnp.searchsorted(T_CB, T) - 1
    T_slice = lax.dynamic_slice_in_dim(T_CB, T_idx, 2, axis = -1)
    c = lax.dynamic_slice_in_dim(c_all, T_idx, 2, axis = -1)
    k = lax.dynamic_slice_in_dim(k_all, T_idx, 2, axis = -1)
    h = lax.dynamic_slice_in_dim(h_all, T_idx, 2, axis = -1)
    a2 = lax.dynamic_slice_in_dim(a2_all, T_idx, 2, axis = -1)
    site_coeffs = lax.dynamic_slice_in_dim(site_all, T_idx, 2, axis = -1)
    attn_coeffs = lax.dynamic_slice_in_dim(attn_all, T_idx, 2, axis = -1)
    sigma_coeffs = lax.dynamic_slice_in_dim(sigma_all, T_idx, 2, axis = -1)
    HW_coeffs = (h, a2)
    return (T_slice, c, k, HW_coeffs, site_coeffs, attn_coeffs, sigma_coeffs)

def f_mag(Mw, c):
    base = c[0] + c[1] * Mw
    c2_term = c[2] * jnp.clip(Mw - 4.5, min = 0)
    c3_term = c[3] * jnp.clip(Mw - 5.5, min = 0)
    c4_term = c[4] * jnp.clip(Mw - 6.5, min = 0)
    return base + c2_term + c3_term + c4_term

def f_geom(Mw, R_rup, 
           c):
    return (c[5] + c[6] * Mw) * jnp.log((R_rup ** 2 + c[7] ** 2) ** (1 / 2))

def f_SOF(Mw, SOF_flag, 
          c):
    SOF_idx = (SOF_flag + 1).astype(jnp.int32)
    c_SOF = lax.select_n(SOF_idx, c[8], c[0] * 0., c[9])
    return c_SOF * jnp.clip(Mw - 4.5, min = 0, max = 1)

def f_HW(Mw, dip, width, R_jb, R_rup, R_x, z_tor,
         c, HW_coeffs):
    h, a2 = HW_coeffs
    R1 = width * jnp.cos(jnp.deg2rad(dip))
    R2 = 62 * Mw - 350
    R1_ratio = R_x / R1
    R_ratio = (R_x - R1) / (R2 - R1)
    f1 = h[1] + h[2] * R1_ratio + h[3] * R1_ratio ** 2
    f2 = h[4] + h[5] * R_ratio + h[6] * R_ratio ** 2

    Rx_cond = (R_x > 0) + (R_x >= R1)
    HW_R_x = lax.select_n(Rx_cond, f1, jnp.clip(f2, min = 0))
    HW_R_rup = lax.select(R_rup > 0, (R_rup - R_jb) / R_rup, 1.)
    HW_Mw = jnp.clip(Mw - 5.5, min = 0, max = 1) * (1 + a2 * (Mw - 6.5))
    HW_z_tor = jnp.clip(1 - 0.06 * z_tor, min = 0)
    HW_dip = (90 - dip) / 45
    return c[10] * HW_R_x * HW_R_rup * HW_Mw * HW_z_tor * HW_dip

def f_site(vs30, pga_rock,
           c, k, site_coeffs):
    C, n = site_coeffs
    vs30_k1 = vs30 / k[1]
    log_vs30_k1 = jnp.log(vs30_k1)
    pga_rock_C = pga_rock + C
    G_less = c[11] * log_vs30_k1 + k[2] * \
            (jnp.log(pga_rock_C * vs30_k1 ** n) - \
             jnp.log(pga_rock_C))
    G_greq = (c[11] + k[2] * n) * log_vs30_k1
    site_G = lax.select(vs30 <= k[1], G_less, G_greq)
    # No Japan
    return site_G

def f_sed(z2p5, 
          c, k):
    # REPLACE WITH NESTED WHERE
    empty = c[0] * 0.
    z2p5_arr = empty + z2p5
    z2p5_cond = (z2p5_arr < 1).astype(int) + (z2p5_arr < 3).astype(int)
    z2p5_less = c[14] * (z2p5 - 1)
    z2p5_gr = c[16] * k[3] * jnp.exp(-0.75) * (1 - jnp.exp(-0.25 * (z2p5 - 3)))
    return lax.select_n(z2p5_cond, z2p5_less, empty, z2p5_gr)

def f_hyp(Mw, z_hyp,
          c):
    hyp_H = jnp.clip(z_hyp - 7, min = 0, max = 13)
    hyp_M = c[17] + (c[18] - c[17]) * jnp.clip(Mw - 5.5, min = 0, max = 1)
    return hyp_H * hyp_M

def f_dip(Mw, dip,
          c):
    return c[19] * dip * jnp.clip(5.5 - Mw, min = 0, max = 1)

def f_attn(R_rup,
           c, attn_coeffs):
    Dc20_CA = attn_coeffs
    return (c[20] * Dc20_CA) * jnp.clip(R_rup - 80, min = 0)

def f_lnSA(Mw, width, dip, z_hyp, z_tor, SOF_flag,
                vs30, z2p5, R_jb, R_rup, R_x,
                c, k, HW_coeffs, site_coeffs, attn_coeffs):
    other_terms = f_mag(Mw, c) + \
                f_geom(Mw, R_rup, c) + \
                f_SOF(Mw, SOF_flag, c) + \
                f_HW(Mw, dip, width,
                        R_jb, R_rup, R_x,
                        z_tor,
                        c, HW_coeffs) + \
                f_hyp(Mw, z_hyp, c) + \
                f_dip(Mw, dip, c) + \
                f_attn(R_rup, c, attn_coeffs)
    site_sed_rock = f_site(1100, 0, c, k, site_coeffs) + f_sed(0.398, c, k)
    pga_rock = jnp.exp(other_terms + site_sed_rock)

    site_sed_else = f_site(vs30, pga_rock, c, k, site_coeffs) + f_sed(z2p5, c, k)
    return other_terms + site_sed_else, pga_rock

def f_sig(Mw, vs30, pga_rock, 
          k, sigma_coeffs, site_coeffs):
    tau1, tau2, phi1, phi2, phi_lnAF, rho = sigma_coeffs
    C, n = site_coeffs
    tau_lny = tau2 + (tau1 - tau2) * jnp.clip(5.5 - Mw, min = 0, max = 1)
    phi_lny = phi2 + (phi1 - phi2) * jnp.clip(5.5 - Mw, min = 0, max = 1)

    alpha = k[2] * pga_rock * (1 / (pga_rock + C * (vs30 / k[1]) ** n) - \
                                    1 / (pga_rock + C))
    alpha = jnp.clip(alpha, min = 0)

    tau_lnPGA = tau_lny[-2]
    tau = (tau_lny ** 2 + alpha ** 2 * tau_lnPGA ** 2 + 2 * alpha * rho * tau_lny * tau_lnPGA)
    phi_lnyB = (phi_lny ** 2 - phi_lnAF ** 2) ** (1 / 2)
    phi_lnPGAB = phi_lnyB[-2]
    phi = (phi_lnyB ** 2 + phi_lnAF ** 2 + alpha ** 2 * phi_lnPGAB ** 2 + 2 * alpha * rho * phi_lnyB * phi_lnPGAB)
    return (tau ** 2 + phi ** 2) ** (1 / 2)

def gmm_CB14(Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
    T_slice, c, k, HW_coeffs, site_coeffs, attn_coeffs, sigma_coeffs = slice_coeffs(T)

    SOF_flag = fault.calc_SOF_flag()
    R_jb, R_rup, R_epi, R_hyp, R_x = R
    
    lnSA, pga_rock = f_lnSA(Mw, fault.width, fault.dip, fault.z_hyp, fault.z_tor, SOF_flag,
                            site.vs30, site.z2p5, 
                            R_jb, R_rup, R_x, 
                            c, k, HW_coeffs, site_coeffs, attn_coeffs)
    std = f_sig(Mw, site.vs30, pga_rock,
                k, sigma_coeffs, site_coeffs)
    lnSA = jnp.interp(T, T_slice, lnSA)
    std = jnp.interp(T, T_slice, std)
    return lnSA, std