import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .gm_utils import *

gmc = pl.read_csv(files("seismic") / "CB14_coeffs.csv")
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T = gmc[0]
T_sort = jnp.argsort(T)
empty = jnp.zeros_like(T, dtype = float)
c = gmc[1:22]
# Region 0 (all others)/1/2/3
Dc20_CA, Dc20_JP, Dc20_CH = gmc[22:25]
a2 = gmc[25]
h = gmc[26:32]
h = jnp.insert(h, 0, empty, axis = 0)
k = gmc[32:35]
k = jnp.insert(k, 0, empty, axis = 0)
C, n = gmc[35:37]
phi1, phi2 = gmc[37:39]
tau1, tau2 = gmc[39:41]
phi_lnAF, phi_C, rho = gmc[41:]
cy_csim = 0.1

def f_mag(Mw):
    base = c[0] + c[1] * Mw
    c2_term = c[2] * jnp.clip(Mw - 4.5, min = 0)
    c3_term = c[3] * jnp.clip(Mw - 5.5, min = 0)
    c4_term = c[4] * jnp.clip(Mw - 6.5, min = 0)
    return base + c2_term + c3_term + c4_term

def f_geom(Mw, R_rup):
    return (c[5] + c[6] * Mw) * jnp.log((R_rup ** 2 + c[7] ** 2) ** (1 / 2))

def f_SOF(Mw, SOF_flag):
    SOF_idx = (SOF_flag + 1).astype(jnp.int32)
    c_SOF = lax.select_n(SOF_idx, c[8], empty, c[9])
    return c_SOF * jnp.clip(Mw - 4.5, min = 0, max = 1)

def f_HW(Mw, dip, width, R_jb, R_rup, R_x, z_tor):
    R1 = width * jnp.cos(jnp.deg2rad(dip))
    R2 = 62 * Mw - 350
    f1 = h[1] + h[2] * (R_x / R1) + h[3] * (R_x / R1) ** 2
    f2 = h[4] + h[5] * ((R_x - R1) / (R2 - R1)) + h[6] * ((R_x - R1) / (R2 - R1)) ** 2
    Rx_cond = (R_x > 0) + (R_x >= R1)
    HW_R_x = lax.select_n(Rx_cond, f1, jnp.clip(f2, min = 0))
    HW_R_rup = lax.select(R_rup > 0, (R_rup - R_jb) / R_rup, 1.)
    HW_Mw = jnp.clip(Mw - 5.5, min = 0, max = 1) * (1 + a2 * (Mw - 6.5))
    HW_z_tor = jnp.clip(1 - 0.06 * z_tor, min = 0)
    HW_dip = (90 - dip) / 45
    return c[10] * HW_R_x * HW_R_rup * HW_Mw * HW_z_tor * HW_dip

def f_site(vs30, pga_rock):
    G_less = c[11] * jnp.log(vs30 / k[1]) + k[2] * \
            (jnp.log(pga_rock + C * (vs30 / k[1]) ** n) - \
             jnp.log(pga_rock + C))
    G_greq = (c[11] + k[2] * n) * jnp.log(vs30 / k[1])
    site_G = lax.select(vs30 <= k[1], G_less, G_greq)
    # No Japan
    return site_G

def f_sed(z2p5):
    z2p5_arr = empty + z2p5
    z2p5_cond = (z2p5_arr < 1).astype(int) + (z2p5_arr < 3).astype(int)
    z2p5_less = c[14] * (z2p5 - 1)
    z2p5_gr = c[16] * k[3] * jnp.exp(-0.75) * (1 - jnp.exp(-0.25 * (z2p5 - 3)))
    return lax.select_n(z2p5_cond, z2p5_less, empty, z2p5_gr)

def f_hyp(Mw, z_hyp):
    hyp_H = jnp.clip(z_hyp - 7, min = 0, max = 13)
    hyp_M = c[17] + (c[18] - c[17]) * jnp.clip(Mw - 5.5, min = 0, max = 1)
    return hyp_H * hyp_M

def f_dip(Mw, dip):
    return c[19] * dip * jnp.clip(5.5 - Mw, min = 0, max = 1)

def f_attn(R_rup):
    return (c[20] * Dc20_CA) * jnp.clip(R_rup - 80, min = 0)

def f_lnSA(Mw, width, dip, z_hyp, z_tor, SOF_flag,
                vs30, z2p5, R_jb, R_rup, R_x):
    other_terms = f_mag(Mw) + \
                f_geom(Mw, R_rup) + \
                f_SOF(Mw, SOF_flag) + \
                f_HW(Mw, dip, width,
                        R_jb, R_rup, R_x,
                        z_tor) + \
                f_hyp(Mw, z_hyp) + \
                f_dip(Mw, dip) + \
                f_attn(R_rup)
    site_sed_rock = f_site(1100, 0) + f_sed(0.398)
    pga_rock = jnp.exp(other_terms + site_sed_rock)

    site_sed_else = f_site(vs30, pga_rock) + f_sed(z2p5)
    return other_terms + site_sed_else, pga_rock

def f_sig(Mw, vs30, pga_rock):
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

def f_CB14(Mw:float, site:Site, fault:Fault):
    R_jb = calc_R_jb(site, fault)
    R_rup = calc_R_rup(site, fault)
    R_x = calc_R_x(site, fault)
    SOF_flag = fault.calc_SOF_flag()
    lnSA, pga_rock = f_lnSA(Mw, fault.width, fault.dip, fault.z_hyp, fault.z_tor, SOF_flag,
                            site.vs30, site.z2p5, 
                            R_jb, R_rup, R_x)
    std = f_sig(Mw, site.vs30, pga_rock)
    lnSA = jnp.interp(T_master, T[T_sort], lnSA[T_sort])
    std = jnp.interp(T_master, T[T_sort], std[T_sort])
    return lnSA, std