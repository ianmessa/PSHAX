# Jaxlib
import jax
from jax import lax
from jax import random as jrnd
from jax import numpy as jnp
from jax import tree_util as jtu
jax.config.update('jax_enable_x64', True)

# Others
from matplotlib import pyplot as plt

# This
from numerics import *
from seismic import *
from hazcalc import *

# Simple site
x_site, y_site = 0., 0.
vs30 = 760
z1p0, z2p5 = 1.3, 0.
site = Site(x_site, y_site, vs30, z1p0, z2p5, 0.)

# Fault 1 is large, faraway earthquakes; fault 2 is small, close
x_fault1, y_fault1 = 50., 49.
theta1, width1 = 225, 1.5
mfd1, M_marg1 = MFD(2.1, 0.9), jnp.array([7., 8.5])
x_fault2, y_fault2 = -20., 15.
theta2, width2 = 30, 2.7
mfd2, M_marg2 = MFD(2.6, 1.8), jnp.array([5., 6.])
# Shared fault params
z_hyp = 1.5
z_tor = 1.
dip = 45
rake = 0.
# Fault 1
fault1 = Fault(x_fault1, x_fault2, z_hyp, z_tor, theta1, dip, rake, width1, 0., mfd1)
fault2 = Fault(x_fault2, y_fault2, z_hyp, z_tor, theta2, dip, rake, width2, 0., mfd2)
faults = [fault1, fault2]
# Scenario magnitude tree
M_margs = jnp.stack([M_marg1, M_marg2])

# GMMs
gmms = [gmm_ASK14, gmm_BSSA14, gmm_CB14, gmm_CY14, gmm_Idriss14]
#c_AAY14 = 1.674
#gmms = gmms + [gmm_epi_AAY14(gmm, c_AAY14) for gmm in gmms] + [gmm_epi_AAY14(gmm, -c_AAY14) for gmm in gmms]
# GMM weights
w_gmms = jnp.ones(len(gmms)) / len(gmms)
gmmlt = GMMLT(gmms, w_gmms)

# Intensity measures + period
n_ims = 10
ims = jnp.logspace(-3, -1, n_ims, base = 10)
# Evaluation im
ime = ims[4]
T = 0.05

# Hazard calculator
dM = 0.04 # (just to speed things up compared to the default 0.01)
haz_calc = HazCalculator(gmmlt, dM)

# With M/lnR
    # ----- KERNEL MATRIX FIGURES -----
    # Enumerate scenario with sobol sequence
    # Produce kernel matrices
    # Save them 

    # For marginal/cumulative:
    #   # 