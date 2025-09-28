# Stolen from USGS github
import jax
from jax import numpy as jnp
from jax import random as jrnd
from jax.tree_util import Partial
from jax.experimental.checkify import check

from collections.abc import Callable

# Get SOF from rake
def f_SOF_flag(rake):
    abs_rake = jnp.abs(rake)
    is_SS = (~jnp.logical_and(abs_rake > 30, abs_rake < 150)).astype(int)
    # [reverse, SS, normal] -> [-1, 0, 1]
    return (is_SS * - jnp.sign(rake)).astype(int)

class gm_scenario:
        def __init__(self, Mw, dip, rake, width, 
                     R_jb, R_rup, R_x, 
                     vs30, 
                     z1p0, z2p5, z_hyp, z_tor,
                     R_y0:int = 0, region:int = 0,
                     HW_flag:bool = False, vs30inf_flag:bool = True):

                """Ground Motion Scenario for an ergodic GMM. Most parameters are intuitive.\\
                        width, R_jb, R_rup, R_x, z1p0, z2p5, z_hyp, z_tor, R_y0 are all in km.
                        HW_flag is only for ASK14. SOF is (-1, 0, 1) for (RV, SS, NML), calculated
                        from input rake. vs30inf_flag is
                        true if vs30 is estimated/inferred. Region is barely used. """

                names = ['Mw', 'dip', 'rake', 'width', 
                         'R_jb', 'R_rup', 'R_x', 
                         'vs30', 'vs30_flag', 
                         'z1p0', 'z2p5', ' z_hyp', 'z_tor',
                         'SOF', 'HW_flag', 'R_y0', 'region']
                
                self.Mw = jnp.array(Mw)
                self.dip = jnp.array(dip)
                self.rake = jnp.array(rake)
                self.width = jnp.array(width)
                self.R_jb = jnp.array(R_jb)
                self.R_rup = jnp.array(R_rup)
                self.R_x = jnp.array(R_x)
                self.vs30 = jnp.array(vs30)
                self.vs30inf_flag = jnp.array(vs30inf_flag)
                self.z1p0 = jnp.array(z1p0)
                self.z2p5 = jnp.array(z2p5)
                self.z_hyp = jnp.array(z_hyp)
                self.z_tor = jnp.array(z_tor)
                self.HW_flag = jnp.array(HW_flag)
                self.R_y0 = jnp.array(R_y0)
                self.region = jnp.array(region)
                self.SOF_flag = f_SOF_flag(rake)

        def __setattr__(self, name, value):
                # Make all float updates jax arrays for vmap compatibility.
                if not isinstance(value, jax.Array):
                        value = jnp.array(value, dtype = float)
                super().__setattr__(name, value)

def f_SA_avg(T:jax.Array, lnSA:jax.Array, 
              delta:float = 0.1):
    """Calculate average spectral acceleration."""
    filter = T > 0
    T_new = jnp.linspace(T[filter].min(), T[filter].max(), delta)
    new_lnSA = jnp.interp(jnp.log(T_new), jnp.log(T[filter]), lnSA[filter])
    new_SA = jnp.exp(new_lnSA)
    return jnp.mean(new_SA)

def f_PGA(T, lnSA):
    """Grab PGA. Formatted same as f_SA_avg for simplicity."""
    return jnp.exp(lnSA[-2])