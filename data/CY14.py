import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jrnd

import polars as pl 

from data.gm_utils import gm_scenario

gmc = pl.read_csv('CY14_coeffs.csv')
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T
T = gmc[0]
empty = jnp.zeros_like(T)

c_RB = gmc[4]
c_n, c_M, c_HM = gmc[jnp.array([12, 13, 16])]
c = jnp.empty((12, 5, T.shape[0]))
# c1, c1a - d
c = c.at[1].set(gmc[7:12])
# c2
c = c.at[2, 0].set(gmc[1])
# c3
c = c.at[3, 0].set(gmc[14])
# c4
c = c.at[4, 0:2].set(gmc[2:4])
# c5 - c7
c = c.at[5:8, 0].set(gmc[jnp.array([15, 17, 18])])
# c8
c = c.at[8, 0:3].set(gmc[jnp.array([5, 6, 20])])
# c9, c9a, c9b
c = c.at[9, 0:3].set(gmc[21:24])
# c11, c11b
c = c.at[11, jnp.array([0, 2])].set(gmc[24:26])
# c_gamma
c_gamma = jnp.insert(gmc[26:29], 0, empty, axis = 0)
phi = jnp.insert(gmc[29:35], 0, empty, axis = 0)
tau = jnp.insert(gmc[35:37], 0, empty, axis = 0)
sigma = jnp.insert(gmc[37:40], 0, empty, axis = 0)
sigma2_JP, gamma_JP_IT, gamma_WN = gmc[40:43]
phi1_JP, phi5_JP, phi6_JP = gmc[43:]

z_tor_const_RV = jnp.array([2.704, 1.226, 5.849])
z_tor_const_NM = jnp.array([2.673, 1.136, 4.97])

