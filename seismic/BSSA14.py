import jax
from jax import lax
from jax import numpy as jnp

import polars as pl
from importlib.resources import files

from .gm_utils import *

gmc = pl.read_csv(files("seismic") / "BSSA14_coeffs.csv")

gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T = gmc[0]
T_sort = jnp.argsort(T)
empty = jnp.zeros_like(T, dtype = float)
e = gmc[1:8]
Mh = gmc[8]
c = gmc[9:12]
c = jnp.insert(c, 0, empty, 0)
M_ref, R_ref, h, Dc3CaTw, Dc3CnTr, Dc3ItJp = gmc[12:18]
c_lin, vc, v_ref, = gmc[18:21]
F = gmc[21:27]
F = jnp.insert(F, 0, empty, axis = 0)
F = jnp.insert(F, 2, empty, axis = 0)
R1, R2 = gmc[27:29]
dPhi_R, dPhi_v = gmc[29:31]
v1, v2, phi1, phi2, tau1, tau2 = gmc[31:]
A4 = 570.94 ** 4
B4 = 1360 ** 4

# Source term
def f_source(Mw, SOF_flag):
    SOF_idx = (SOF_flag + 1).astype(jnp.int32)
    e_SOF = lax.select_n(SOF_idx, e[3], e[1], e[2])
    MwMh = Mw - Mh
    e_addn = lax.select(Mw <= Mh, e[4] * MwMh + e[5] * MwMh ** 2, e[6] * MwMh)
    return e_SOF + e_addn

def f_path(Mw, R):
    return jnp.log(R / R_ref) * (c[1] + c[2] * (Mw - M_ref)) + (c[3] + Dc3CaTw) * (R - R_ref)

# Site term
def f_site(vs30, z1p0, PGA_rock):
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
    F_dz1 = lax.select(filter2, empty, F_dz1)

    return ln_Flin + ln_Fnl + F_dz1

def f_lnSA(Mw, SOF_flag, 
           vs30, z1p0, 
           R_jb):
    R = (R_jb ** 2 + h ** 2) ** (1 / 2)
    # Add index to select PGA?...
    PGA_rock = jnp.exp(f_source(Mw, SOF_flag) + f_path(Mw, R))
    
    return f_source(Mw, SOF_flag) + \
           f_path(Mw, R) + \
           f_site(vs30, z1p0, PGA_rock)

def f_sigma(Mw, vs30, R_jb):
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

def f_BSSA14(Mw:float, site:Site, fault:Fault):
    R_jb = calc_R_jb(site, fault)
    SOF_flag = fault.calc_SOF_flag()
    lnSA = f_lnSA(Mw, SOF_flag, site.vs30, site.z1p0, R_jb)
    std = f_sigma(Mw, site.vs30, R_jb)
    lnSA = jnp.interp(T_master, T[T_sort], lnSA[T_sort])
    std = jnp.interp(T_master, T[T_sort], std[T_sort])
    return lnSA, std