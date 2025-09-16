# Stolen from USGS github
import jax
from jax import numpy as jnp
from jax import random as jrnd
from jax.tree_util import Partial

from collections.abc import Callable

class gm_scenario:
        def __init__(self, Mw, dip, rake, width, 
                     R_jb, R_rup, R_x, 
                     vs30, vs30_flag,
                     z1p0, z2p5, z_hyp, z_tor,
                     HW_flag: int = 0, R_y0:int = 0, region:int = 0):

                "Ground Motion Scenario for an ergodic GMM."

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
                self.HW_flag = jnp.array(HW_flag)
                self.R_y0 = jnp.array(R_y0)
                self.region = jnp.array(region)

        def __setattr__(self, name, value):
                # Make all float updates jax arrays for vmap compatibility.
                if not isinstance(value, jax.Array):
                        value = jnp.array(value, dtype = float)
                super().__setattr__(name, value)

class gm_tree:
        def __init__(self, T_all:jax.Array, *gmms):
                """Logic tree of ground motion models where T_all is 
                a universal set of periods to which all models will be log-log interpolated."""
                self.gmms = gmms
                self.T_all = T_all
                self.size = len(gmms)
        def __call__(self, weights:jax.Array, scn:gm_scenario, 
                     full_output: bool = False):
                """Evaluate tree for a certain ground motion scenario and set of weights.
                   We include the weight array in the call function to make Monte Carlo sampling easier."""
                assert weights.shape[0] == self.size, "Must have one weight for each ground motion model!"
                assert weights.sum() == 1, "Weights must sum to 1!"
                lnSAs = []
                sigmas = []
                for i,gmm in enumerate(self.gmms):
                        T, lnSA, sigma = gmm(scn)
                        if self.T_all is None: 
                                self.T_all = T
                        lnSA_new, sigma_new = [jnp.interp(jnp.log(self.T_all), jnp.log(T), val) for val in [lnSA, sigma]]
                        lnSAs.append(lnSA_new)
                        sigmas.append(sigma_new)
                lnSAs, sigmas = jnp.array([lnSAs]), jnp.array([sigmas])
                if full_output:
                        return lnSAs, sigmas
                else:
                        return lnSAs @ weights, ((sigmas ** 2) @ weights) ** (1 / 2)

def fn_SA_avg(T:jax.Array, lnSA:jax.Array, 
              delta:float = 0.1):
    """Calculate average spectral acceleration."""
    filter = T > 0
    T_new = jnp.linspace(T[filter].min(), T[filter].max(), delta)
    new_lnSA = jnp.interp(jnp.log(T_new), jnp.log(T[filter]), lnSA[filter])
    new_SA = jnp.exp(new_lnSA)
    return jnp.mean(new_SA)

def fn_PGA(T, lnSA):
    """Grab PGA. Formatted same as fn_SA_avg for simplicity."""
    return jnp.exp(lnSA[-2])

def monte_carlo(tree:gm_tree, scn:gm_scenario, 
                num_trials:int,
                key_num:int = 0):
    """Monte Carlo sampling a tree of GMMs. Returns a (num_trials * tree.T_all) array."""
    key = jrnd.key(key_num)
    weights_arr = jrnd.ball(key, tree.size, 1, (num_trials,))
    weights_arr = weights_arr / weights_arr.sum(axis = -1)[:, None]
    lnSAs, sigmas = jax.vmap(Partial(tree, scn)(weights_arr))
    return lnSAs, sigmas

def fn_V(tree:gm_tree, weights:jax.Array, 
         scn_1:gm_scenario, scn_2:gm_scenario,
         fn_SA_metric:Callable = fn_PGA):
    """Calculate correlation between two scenarios according to AL21 using
    PGA or SA_avg (latter described in O'Reilly 21)."""
    assert fn_SA_metric in [fn_PGA, fn_SA_avg], "Need to use PGA or SA average."
    lnSAs_1, sigmas_1 = tree(scn_1, weights, full_output = True)
    lnSAs_2, sigmas_2 = tree(scn_2, weights, full_output = True)
    SA_1, SA_2 = fn_SA_metric(lnSAs_1), fn_SA_metric(lnSAs_2)
    V = (weights * (SA_1 - SA_2) ** 2).sum() / (2 * tree.size)
    return V

def gm_default():
        return gm_scenario(6.8, 75, 30, 3, 
                           2.5, 2.5, 2.5, 
                           760, 0,
                           -.5, -.5, 0.5, 0,
                           0, 0, 0)