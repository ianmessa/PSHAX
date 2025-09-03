# Stolen from USGS github
from jax import numpy as jnp
class gm_scenario:
        def __init__(self, Mw, dip, rake, width, 
                     R_jb, R_rup, R_x, 
                     vs30, vs30_flag,
                     z1p0, z2p5, z_hyp, z_tor,
                     SOF:int = 0, HW_flag: int = 0, R_y0:int = 0, region:int = 0):
                
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
                self.SOF = jnp.array(SOF)
                self.HW_flag = jnp.array(HW_flag)
                self.R_y0 = jnp.array(R_y0)
                self.region = jnp.array(region)