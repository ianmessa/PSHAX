import jax
from jax import lax
from jax import numpy as jnp

import polars as pl

from gm_scenario import *

gmc = pl.read_csv('BSSA14_coeffs.csv')

gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
print(gmc_col)
gmc = gmc.cast(pl.Float64).to_jax().T

T = gmc[0]
empty = jnp.zeros_like(T, dtype = float)
e = gmc[1:8]
Mh = gmc[8]
c = gmc[9:12]
c = jnp.insert(c, 0, empty, 0)
M_ref, R_ref, h, Dc3CaTw, Dc3CnTr, Dc3ItJp = gmc[12:18]
c_lin, Vc, V_ref, = gmc[18:21]
f = gmc[21:27]
f = jnp.insert(f, 0, empty, axis = 0)
f = jnp.insert(f, 2, empty, axis = 0)
R1, R2 = gmc[27:29]
dPhi_R, dPhi_V = gmc[29:31]
v1, v2, phi1, phi2, tau1, tau2 = gmc[31:]

# Source model (eqn. 2)
def f_event(Mw, SOF):
    # Select last term
    e_last = lax.select(Mw <= Mh, 
                        e[4] * (Mw - Mh) + e[5] * (Mw - Mh) ** 2,
                        e[6] * (Mw - Mh))
    
    # Build SOF condition ([-1, 0, 1] -> [0, 1, 2])
    #   We assume it will always be known for the sake of 
    #   compatibility with other GMPEs. 
    cond_SOF = (SOF + 1).astype(int)
    e_SOF = lax.select_n(cond_SOF, e[3], e[1], e[2])
    return e_SOF#e_SOF + e_last

# Path model (eqns. 3, 4)
def f_path(Mw, R_jb, region):
    R = (R_jb ** 2 + h ** 2) ** (1 / 2)
    cond = 0
    Dc3 = lax.select_n(cond, empty, Dc3CaTw, Dc3CnTr, Dc3ItJp)
    return jnp.log(R / R_ref) * (c[1] + c[2] * (Mw - M_ref)) + (c[3] + Dc3) * (R - R_ref)

# Site model (eqn. 5 - 12): 
def f_site(Mw, R_jb, vs30, z1p0, region, PGA_rock):
    vs30_capped = jnp.clip(vs30, max = Vc)
    lnF_lin = c_lin * jnp.log(vs30_capped / V_ref)
    
    # Nonlinear scaling
    f2 = f[4] * (jnp.exp(f[5] * (vs30_capped - 360)) - \
                 jnp.exp(f[5] * (760 - 360)))
    lnF_nl = f[1] + f2 * jnp.log((PGA_rock + f[3]) / f[3])

    # Basin amplification
    mu_CA = jnp.exp(-7.15 / 4 * jnp.log((vs30 ** 4 + 570.94 ** 4) / (1360 ** 4 + 570.94 ** 4))) / 1000
    mu_JP = jnp.exp(-5.23 / 2 * jnp.log((vs30 ** 2 + 412.39 ** 2) / (1360 ** 2 + 412.39 ** 2))) / 1000
    mu_z1p0 = lax.select(region == 10, mu_JP, mu_CA)
    dz1 = z1p0 - mu_z1p0

    filt1 = jnp.logical_and(T >= 0.65, dz1 <= (f[7] / f[6]))
    filt2 = jnp.logical_and(T >= 0.65, dz1 > (f[7] / f[6]))
    F_dz1 = jnp.zeros_like(T)
    F_dz1 = F_dz1.at[filt1].set((f[6] * dz1)[filt1])
    F_dz1 = F_dz1.at[filt2].set(f[7, filt2])

    return lnF_lin + lnF_nl + F_dz1

# lnSA model
def f_lnSA(Mw, R_jb, vs30, SOF, z1p0, region, PGA_rock): 

    return f_event(Mw, SOF) + \
             f_path(Mw, R_jb, region) + \
             f_site(Mw, R_jb, vs30, z1p0, region, PGA_rock)

# Aleatory (eqn. 13 - 17)
def f_sigma(Mw, R_jb, vs30):
    # eqn. 14
    tau = tau1 + (tau2 - tau1) * jnp.clip(Mw - 4.5, min = 0,
                                                    max = 1)
    
    # eqn. 17
    phi_Mw = phi1 + (phi2 - phi1) * jnp.clip(Mw - 4.5, min = 0,
                                                      max = 1)
    # eqn.16
    dPhi_R_factor = jnp.clip(jnp.log(R_jb / R1) / jnp.log(R2 / R1), min = 0, 
                                                                  max = 1)
    phi_R_jb = phi_Mw + dPhi_R * dPhi_R_factor
    # eqn. 15
    dPhi_V_factor = jnp.clip(jnp.log(v2 / vs30) / jnp.log(v2 / v1), min = 0,
                                                                  max = 1)
    phi = phi_R_jb + dPhi_V * dPhi_V_factor

    # eqn. 13
    return (phi ** 2 + tau ** 2) ** (1 / 2)

# Put it all together.
def BSSA14(scn:gm_scenario):
    # -2 index is for PGA
    PGA_rock = jnp.exp(f_event(scn.Mw, scn.SOF) + \
                       f_path(scn.Mw, scn.R_jb, scn.region))[-2]
    lnSA = f_lnSA(scn.Mw, scn.R_jb,
                  scn.vs30, 
                  scn.SOF, scn.z1p0, scn.region,
                  PGA_rock)
    sigma = f_sigma(scn.Mw, scn.R_jb, scn.vs30)

    return T, lnSA, sigma
