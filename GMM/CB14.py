import jax
from jax import lax
from jax import numpy as jnp

import polars as pl

from gm_scenario import *

gmc = pl.read_csv('CB14_coeffs.csv')
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

T = gmc[0]
empty = jnp.zeros_like(T, dtype = float)
c = gmc[1:22]
# Region 0 (all others)/1/2/3
Dc20_CA, Dc20_JP, Dc20_CH = gmc[22:25]
a2 = gmc[25]
h = gmc[26:32]
h = jnp.insert(h, 0, empty, axis = 0)
k = gmc[32:35]
k = jnp.insert(k, 0, empty, axis = 0)
c_SA, n = gmc[35:37]
phi_c = gmc[37:39]
tau_c = gmc[39:41]
phi_lnAF, phi_C, rho = gmc[41:]

# Magnitude term (eqn. 2)
def f_mag(Mw):
    # Straightfwd
    ret = c[0] + \
        c[1] * Mw + \
        c[2] * jnp.clip(Mw - 4.5, min = 0) + \
        c[3] * jnp.clip(Mw - 5.5, min = 0) + \
        c[4] * jnp.clip(Mw - 6.5, min = 0)
    return ret

# Attenuation term (eqn. 3)
def f_dis(Mw, R_rup):
    # Straightfwd
    return (c[5] + c[6] * Mw) * jnp.log((R_rup ** 2 + c[7] ** 2) ** (1/2))

# Fault style term (eqn. 4 - 6)
def f_SOF(Mw, SOF):
    # [-1, 0, 1] -> [0, 1, 2]
    cond = (SOF + 1).astype(int)
    # Select constants, multiply by clipped fault fn
    c_SOF = lax.select_n(cond, c[8], empty, c[9])
    return c_SOF * jnp.clip(Mw - 4.5, min = 0, max = 1)

# Hanging wall (eqns. 7 - 16)
def f_hng(Mw, dip, width, R_jb, R_rup, R_x, z_tor): 
    # Implemented in order of appearance in scenario class
    hng_Mw = jnp.clip(Mw - 5.5, min = 0, max = 1) * (1 + a2 * (Mw - 6.5))
    hng_dip = (90 - dip) / 45
    hng_R_rup = lax.select(R_rup == 0, jnp.ones_like(R_rup), (R_rup - R_jb) / R_rup)

    # R_x term is a pain
    R1 = width * jnp.cos(dip)
    R2 = 62 * Mw - 350
    R_x1 = h[1] + h[2] * (R_x / R1) + h[3] * (R_x / R1) ** 2
    R_x2 = h[4] + h[5] * ((R_x - R1) / (R2 - R1)) + h[6] * ((R_x - R1) / (R2 - R1)) ** 2
    R_x2 = jnp.clip(R_x2, min = 0)
    # Slam together
    cond = (R_x >= 0).astype(int) + (R_x >= R1).astype(int)
    hng_R_x = lax.select_n(cond, empty, R_x1, R_x2)

    hng_Z_tor = jnp.clip(1 - 0.06 * z_tor, min = 0)

    return c[10] * hng_Mw * hng_dip * hng_R_rup * hng_R_x * hng_Z_tor

# Site response (eqns. 17 - 19)
def f_site(vs30, SA1100, region): 
    s_J = int(region == 2)
    G_lo = c[11] * jnp.log(vs30 / k[1]) + k[2] * \
        (jnp.log(SA1100 + c_SA * (vs30 / k[1]) ** n) - \
         jnp.log(SA1100 + c_SA))
    G_hi = (c[11] + k[2] * n) * jnp.log(vs30 / k[1])
    G = lax.select(vs30 <= k[1], G_lo, G_hi)
    
    c_J = lax.select(vs30 <= 200, c[12], c[13])
    coeff_minus_J = lax.select(vs30 <= 200, jnp.log(200 / k[1]), empty)
    J = (c_J + k[2] * n) * (jnp.log(vs30 / k[1]) - coeff_minus_J)
    return G + s_J * J

# Basin depth term (eqn. 20)
def f_sed(z2p5, region):
    s_J = int(region == 2)
    ret_lo = (c[14] + c[15] * s_J * (z2p5 - 1))
    ret_hi = c[16] * k[3] * jnp.exp(-0.75) * \
            1 - jnp.exp(-0.25 * (z2p5 - 3))
    
    cond = int(z2p5 > 1 and z2p5 <= 3) + int(z2p5 > 3)

    return lax.select_n(cond, ret_lo, empty, ret_hi)

# Hypocentral depth term (eqns. 21 - 23)
def f_hyp(Mw, z_hyp):
    hyp_H = jnp.clip(z_hyp - 13, 
                    min = 0, 
                    max = 13)
    hyp_M = c[17] + (c[18] - c[17]) * jnp.clip(Mw - 5.5,
                                                min = 0,
                                                max = 1)
    
    return hyp_H * hyp_M

# Dip term (eqn. 24)
def f_dip(Mw, dip):
    return c[19] * dip * jnp.clip(5.5 - Mw, 
                                 min = 0, 
                                 max = 1)

# Anelastic attenuation (eqn. 25)
def f_attn(R_rup, region):
    # Select delta C20 by region
    Dc_20 = lax.select_n(region, empty, Dc20_CA, Dc20_CH, Dc20_JP)
    return jnp.clip((c[20] + Dc_20) * (R_rup - 80), 
                    min = 0)

# Full lnSA model (eqn. 1)
def f_lnSA(Mw, dip, width,
           R_jb, R_rup, R_x, 
           vs30,
           z2p5, z_hyp, z_tor, 
           SA_rock, SOF, region):
    return f_mag(Mw) + \
            f_dis(Mw, R_rup) + \
            f_SOF(Mw, SOF) + \
            f_hng(Mw, dip, width,
                  R_jb, R_rup, R_x, 
                  z_tor) + \
            f_site(vs30, SA_rock, region) + \
            f_sed(z2p5, region) + \
            f_hyp(Mw, z_hyp) + \
            f_dip(Mw, dip) + \
            f_attn(R_rup, region)

# Aleatory variability (eqns. 26 - 32)
def f_sigma(Mw, vs30, SA1100):
    # Alpha coeff first
    vs30_arr = jnp.full_like(empty, vs30)
    alpha_nonzero = k[2] * SA1100 * \
                    (1 / (SA1100 + phi_C * (vs30 / k[1]) ** n) - \
                    1 / (SA1100 + phi_C))
    alpha = lax.select(vs30_arr < k[1], alpha_nonzero, empty)

    # Between-event
    tau_lnY = tau_c[2] + (tau_c[1] - tau_c[2]) * jnp.clip(5.5 - Mw, 
                                                          min = 0,
                                                          max = 1)
    tau_lnPGA = tau_lnY[-2]
    tau = jnp.sqrt(tau_lnY ** 2 + \
                   alpha ** 2 * tau_lnPGA ** 2 + \
                    2 * alpha * rho * tau_lnY * tau_lnPGA)

    # Within-event
    phi_lnY = phi_c[2] + (phi_c[1] - phi_c[2]) * jnp.clip(5.5 - Mw, 
                                                          min = 0,
                                                          max = 1)
    phi_lnPGA = phi_lnY[-2]
    phi_lnYB = (phi_lnY ** 2 - phi_lnAF ** 2) ** (1 / 2)
    phi_lnPGAB = (phi_lnPGA ** 2 - phi_lnAF ** 2) ** (1 / 2)
    phi = jnp.sqrt(phi_lnYB ** 2 + phi_lnAF ** 2 + \
                   alpha ** 2 * phi_lnPGAB ** 2 + 
                   2 * alpha * rho * phi_lnYB * phi_lnPGAB)
    
    sigma = (tau ** 2 + phi ** 2) ** (1 / 2)

    return sigma

def CB14(scn: gm_scenario):
    # SA1100
    lnSA1100 = f_lnSA(scn.Mw, scn.dip, scn.width,
                      scn.R_jb, scn.R_rup, scn.R_x,
                      1100,
                      -1., scn.z_hyp, scn.z_tor, 
                      empty, scn.SOF, scn.region)
    SA1100 = jnp.exp(lnSA1100)

    # lnSA
    lnSA = f_lnSA(scn.Mw, scn.dip, scn.width,
                  scn.R_jb, scn.R_rup, scn.R_x,
                  scn.vs30,
                  scn.z2p5, scn.z_hyp, scn.z_tor,
                  SA1100, scn.SOF, scn.region)
    
    # Aleatoric uncertainty
    sigma = f_sigma(scn.Mw, scn.vs30, SA1100)

    # We made it.
    return T, lnSA, sigma