import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .gm_utils import *

gmc = pl.read_csv(files("seismic") / "BSSA14_coeffs.csv")

gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc = gmc.with_columns([pl.col("T").cast(pl.Float64)])
gmc = gmc.sort('T')
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T_BSSA = gmc[0]
empty_all = jnp.zeros_like(T_BSSA, dtype = float)
e_all = gmc[1:8]
# This wasn't here before. May cause problems, must unit test against API.
#   Again.
e_all = jnp.concat([empty_all[None], e_all], axis = 0)
Mh_all = gmc[8]
c_all = gmc[9:12]
c_all = jnp.concat([empty_all[None], c_all], axis = 0)
M_ref_all, R_ref_all, h_all = gmc[12:15]
Dc3CaTw_all, Dc3CnTr_all, Dc3ItJp_all = gmc[15:18]
c_lin_all, vc_all, v_ref_all = gmc[18:21]
F_all = gmc[21:27]
F_all = jnp.insert(F_all, 0, empty_all, axis = 0)
F_all = jnp.insert(F_all, 2, empty_all, axis = 0)
R1_all, R2_all = gmc[27:29]
dPhi_R_all, dPhi_v_all = gmc[29:31]
v1_all, v2_all, phi1_all, phi2_all, tau1_all, tau2_all = gmc[31:]

others_all = jnp.stack([Mh_all, M_ref_all, R_ref_all, h_all,
                          Dc3CaTw_all, Dc3CnTr_all, Dc3ItJp_all,
                          c_lin_all, vc_all, v_ref_all,
                          R1_all, R2_all, dPhi_R_all, dPhi_v_all,
                          v1_all, v2_all, phi1_all, phi2_all, tau1_all, tau2_all], axis = 0)

A4 = 570.94 ** 4
B4 = 1360 ** 4

def slice_coeffs(T):
    T_idx = jnp.searchsorted(T_BSSA, T) - 1
    T_slice = lax.dynamic_slice_in_dim(T_BSSA, T_idx, 2, axis = -1)
    e = lax.dynamic_slice_in_dim(e_all, T_idx, 2, axis = -1)
    c = lax.dynamic_slice_in_dim(c_all, T_idx, 2, axis = -1)
    F = lax.dynamic_slice_in_dim(F_all, T_idx, 2, axis = -1)
    others = lax.dynamic_slice_in_dim(others_all, T_idx, 2, axis = -1)
    Mh, M_ref, R_ref, h, Dc3CaTw, Dc3CnTr, Dc3ItJp = others[:7]
    c_lin, vc, v_ref, R1, R2, dPhi_R, dPhi_v = others[7:14]
    v1, v2, phi1, phi2, tau1, tau2 = others[14:]
    return (T_slice, (e, Mh), (c, M_ref, R_ref, Dc3CaTw), (F, vc, c_lin, v_ref), h, (tau1, tau2, phi1, phi2, dPhi_R, dPhi_v, R1, R2, v1, v2))

# Source term
def f_source(Mw, SOF_flag, source_coeffs):
    e, Mh = source_coeffs
    rv_filt = SOF_flag == -1
    ss_filt = SOF_flag == 0
    nm_filt = SOF_flag == 1
    e_SOF = rv_filt * e[3] + ss_filt * e[1] + nm_filt * e[2]
    MwMh = Mw - Mh
    e_addn = jnp.where(MwMh <= 0, e[4] * MwMh + e[5] * MwMh ** 2, e[6] * MwMh)
    return e_SOF + e_addn

def f_path(Mw, R, path_coeffs):
    c, M_ref, R_ref, Dc3CaTw = path_coeffs
    return jnp.log(R / R_ref) * (c[1] + c[2] * (Mw - M_ref)) + (c[3] + Dc3CaTw) * (R - R_ref)

# Site term
def f_site(vs30, z1p0, PGA_rock, T, site_coeffs):
    F, vc, c_lin, v_ref = site_coeffs
    vs_lin = jnp.clip(vs30, max = vc)
    ln_Flin = c_lin * jnp.log(vs_lin / v_ref)

    vs_clip = jnp.clip(vs30, max = 760.) 
    F2 = F[4] * (jnp.exp(F[5] * (vs_clip - 360.)) - jnp.exp(F[5] * (760. - 360.)))
    ln_Fnl = F[1] + F2 * jnp.log((PGA_rock + F[3]) / F[3])

    z1_ref = jnp.exp(-7.15 / 4. * jnp.log((vs30 ** 4 + A4) / B4)) / 1000
    dz1 = z1p0 - z1_ref
    F_dz1 = F[6] * dz1
    # Cap at F7
    filter1 = dz1 > (F[7] / F[6])
    F_dz1 = lax.select(filter1, F[7], F_dz1)
    # Turn low periods to zero
    filter2 = T < 0.65
    F_dz1 = jnp.where(filter2, 0., F_dz1)

    return ln_Flin + ln_Fnl + F_dz1

def f_lnSA(Mw, T, SOF_flag, 
           vs30, z1p0, 
           R_jb, 
           source_coeffs,
           path_coeffs,
           site_coeffs,
           h):
    R = (R_jb ** 2 + h ** 2) ** (1 / 2)
    # Add index to select PGA?...
    PGA_rock = jnp.exp(f_source(Mw, SOF_flag, source_coeffs) + f_path(Mw, R, path_coeffs))
    
    return f_source(Mw, SOF_flag, source_coeffs) + \
           f_path(Mw, R, path_coeffs) + \
           f_site(vs30, z1p0, PGA_rock, T, site_coeffs)

def f_sigma(Mw, vs30, R_jb,
            sigma_coeffs):
    tau1, tau2, phi1, phi2, dPhi_R, dPhi_v, R1, R2, v1, v2 = sigma_coeffs
    tau = jnp.clip(tau1 + (tau2 - tau1) * (Mw - 4.5), 
                   min = tau1, max = tau2)
    phi = jnp.clip(phi1 + (phi2 - phi1) * (Mw - 4.5), 
                   min = phi1, max = phi2)

    filter_R = R_jb > R1
    coeff_R = jnp.clip((jnp.log(R_jb / R1) - jnp.log(R2 / R1)), max = 1)
    phi_R_mod = phi + dPhi_R * coeff_R
    phi = lax.select(filter_R, phi_R_mod, phi)

    filter_v = vs30 < v2
    coeff_v = jnp.clip(jnp.log(v2 / vs30) / jnp.log(v2 / v1), max = 1)
    phi_v_mod = phi + dPhi_v + coeff_v
    phi = lax.select(filter_v, phi_v_mod, phi)
    return (tau ** 2 + phi ** 2) ** (1 / 2)

def f_BSSA14(Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
    T_slice, source_coeffs, path_coeffs, site_coeffs, h, sigma_coeffs = slice_coeffs(T)

    SOF_flag = fault.calc_SOF_flag()
    R_jb, R_rup, R_epi, R_hyp, R_x = R

    lnSA = f_lnSA(Mw, T_slice, SOF_flag, site.vs30, site.z1p0, R_jb,
                  source_coeffs, path_coeffs, site_coeffs, h)
    std = f_sigma(Mw, site.vs30, R_jb, 
                  sigma_coeffs)
    lnSA = jnp.interp(T, T_slice, lnSA)
    std = jnp.interp(T, T_slice, std)
    return lnSA, std