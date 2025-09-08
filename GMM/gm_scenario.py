# Stolen from USGS github
import jax
from jax import numpy as jnp

# Convert rake to SOF and vice versa
def SOF_from_rake(rake):
        # -1 for reverse, 0 for SS, 1 for normal
        return jnp.sin(jnp.deg2rad(rake)).round().astype(int)

class gm_scenario:
        def __init__(self, Mw, dip, rake, width, 
                     R_jb, R_rup, R_x, 
                     vs30, vs30_flag,
                     z1p0, z2p5, z_hyp, z_tor,
                     HW_flag: int = 0, R_y0:int = 0, region:int = 0):
                
                """N.B. SOF takes on [-1, 0, 1, 2] for [reverse, SS, normal, unlabeled]"""

                names = ['Mw', 'dip', 'rake', 'width', 
                         'R_jb', 'R_rup', 'R_x', 
                         'vs30', 'vs_flag', 
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
                self.vs30_flag = jnp.array(vs30_flag)
                self.z1p0 = jnp.array(z1p0)
                self.z2p5 = jnp.array(z2p5)
                self.z_hyp = jnp.array(z_hyp)
                self.z_tor = jnp.array(z_tor)
                self.SOF = SOF_from_rake(rake)
                self.HW_flag = jnp.array(HW_flag)
                self.R_y0 = jnp.array(R_y0)
                self.region = jnp.array(region)

        def __setattr__(self, name, value):
                if not isinstance(value, jax.Array):
                        value = jnp.array(value, dtype = float)
                super().__setattr__(name, value)

def gm_default():
        return gm_scenario(6.8, 75, 30, 3, 
                           2.5, 2.5, 2.5, 
                           760, False,
                           -1., -1., 0.5, 0,
                           0, 0, 0)

def gm_random():
        pass