# Stolen from USGS github
import typing
from jax import Array
names = ['Mw', 'dip', 'rake', 'width', 'R_jb', 'R_rup', 'R_x', 'vs30', 'vs_flag', 'z1p0', 'z2p5', ' z_hyp', 'z_tor']
def pack_scenario(Mw:float, dip:float, rake:float, width:float,
                  R_jb:float, R_rup:float, R_x:float,
                  vs30:float, vs_flag:bool,
                  z1p0:float, z2p5:float, 
                  z_hyp:float, z_tor:float,
                  names:list = names):
        return dict(zip(names, locals()))