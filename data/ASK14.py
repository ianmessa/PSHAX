import jax
from jax import lax
from jax.tree_util import Partial
from jax import numpy as jnp

import polars as pl

from data.gm_utils import *

##### GROUND MOTION COEFFICIENTS #####
gmc = pl.read_csv('ASK14_coeffs.csv')
gmc[-2, 'T'] = -1.
gmc[-1, 'T'] = -2.
gmc_col = gmc.columns
gmc = gmc.cast(pl.Float64).to_jax().T

# First few
T, v_lin, b, n, M1, c, c4 = gmc[:7]
M = jnp.empty((3, T.shape[0]))
M = M.at[1].set(M1)
M = M.at[2].set(5.0)
# Update M with constant for M2 and 0-row for M0
empty = jnp.zeros_like(T)

# Get a. But they're all out of order...
a = gmc[7:38]
a_idcs = jnp.array([int(ai[1:]) for ai in gmc_col[7:38]])
a = a[jnp.argsort(a_idcs)]
a_missing = jnp.arange(a_idcs.max())
a_missing = a_missing[~jnp.isin(a_missing, a_idcs)]
# Update indices to account for insertions
a_missing = a_missing - jnp.arange(a_missing.shape[0])
a = jnp.insert(a, a_missing, empty, axis = 0)

# Grab s
s_est = gmc[38:40]
s_est = jnp.insert(s_est, 0, empty, axis = 0)
s = gmc[40:]
s = jnp.insert(s, 0, empty, axis = 0)

# Also defining v1 so we don't need to do it twice
v1 = jnp.exp(-0.35 * jnp.log(T / 0.5) + jnp.log(1500))
v1 = v1.at[T <= 0.5].set(1500)
v1 = v1.at[T >= 3].set(800)