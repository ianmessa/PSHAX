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

from jax.scipy.stats.norm import cdf as norm_cdf

def _faults_to_tree(faults:list[Fault])->Fault:
    return jt.map(lambda *xs: jnp.stack(xs), *faults)

@jtu.register_pytree_node_class
class HazCalculator:
    """
    Probabilistic Seismic Hazard Analysis (PSHA) Calculator

    This class provides methods for calculating seismic hazard using a logic tree of Ground Motion Models (GMMs).
    It supports incremental and marginal hazard calculations, ground motion evaluation, and epistemic uncertainty
    quantification for PSHA applications.

    Attributes
    ----------
    gmmlt : GMMLT
        The logic tree of Ground Motion Models (GMMs) with associated weights.
    dM : float, optional
        The magnitude bin size for discretizing the magnitude-frequency distribution (default is 0.01).

    Methods
    -------
    haz_calc_n_M_incr(scn: Scenario)
        Computes incremental rupture rates (number of events per bin) for all faults and magnitude bins.
    mhaz_calc_n_M_incr(M: float, scn: Scenario)
        Computes marginal incremental rupture rates for a specific magnitude M.
    haz_calc_all_lnSA(scn: Scenario)
        Calculates the mean and standard deviation of the natural logarithm of spectral acceleration (lnSA)
        for all GMMs, faults, and magnitude bins.
    mhaz_calc_all_lnSA(M: float, scn: Scenario)
        Calculates the mean and standard deviation of lnSA for all GMMs and faults at a specific magnitude M.
    haz_calc_haz_from_lnSA(im: jax.Array, all_mu_lnSA: jax.Array, all_std_lnSA: jax.Array, all_w: jax.Array, n_M_incr: jax.Array)
        Computes the total hazard curve (annual frequency of exceedance) for a given intensity measure (IM)
        using the computed lnSA statistics and incremental rates.
    haz_calc_epi_from_lnSA(im: jax.Array, all_mu_lnSA: jax.Array, all_std_lnSA: jax.Array, all_w: jax.Array, n_M_incr: jax.Array, ddof: int = 0)
        Computes epistemic uncertainty (weighted standard deviation) in the hazard curve due to logic tree weights.
    tree_flatten()
        Flattens the HazCalculator object for JAX pytree compatibility.
    tree_unflatten(aux, children)
        Reconstructs a HazCalculator object from flattened components for JAX pytree compatibility.
    """
    def __init__(self, gmmlt:GMMLT, dM:float = 0.01):
        """Initialize with GMMs and a bin size"""
        self.gmmlt = gmmlt
        self.dM = dM

    #### HELPERS ####
    def _calc_M_n_M_bins(self, fault_tree:Fault, M_ranges:jax.Array|None = None):
        # Magnitude bins.
        M_min = fault_tree.mfd.M_min if M_ranges is None else M_ranges[:,0]
        M_max = fault_tree.mfd.M_max if M_ranges is None else M_ranges[:,1]
        bins_M = jnp.arange(M_min.min(), M_max.max() + self.dM, self.dM)
        # MFD exceedance rates
        n_M_exc = jax.vmap(fault_tree.mfd.calc_lmdaM)(bins_M)
        # MFD incremental rates
        n_M_incr = n_M_exc[:-1] - n_M_exc[1:]
        # Roots for GMM evaluation.
        roots_M = (bins_M[:-1] + bins_M[1:]) / 2
        # Array of ones/zeros for each fault signifying array inside/outside range
        weights_mask = (roots_M[:, None] > M_min[None, :]) & (roots_M[:, None] < M_max[None, :])
        # We can think of our incremental rates as already including quadrature weights, so 
        #   we'll just multiply them by the mask.
        n_M_incr = n_M_incr * weights_mask
        # Mask out values that aren't selected for either fault
        M_mask = jnp.logical_or(*weights_mask.T)
        return roots_M, n_M_incr, M_mask
    
    def _calc_exc_im(self, im:float, mu_lnSA:jax.Array, std_lnSA:jax.Array):
        # Calculate cumulative probability
        cum_prob_im = norm_cdf(jnp.log(im), loc = mu_lnSA, scale = std_lnSA)
        # Note that we SHOULD implement a 3-sigma upper bound here, but it appears to be
        #   incompatible with jax.jet...
        return 1 - cum_prob_im

    #### HAZARD CALC ####
    def calc_haz(self, im:jax.Array, T:float, site:Site, faults:list[Fault], M_ranges:jax.Array|None = None):
        # Get rates, bins, roots
        fault_tree = _faults_to_tree(faults)
        roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
        roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
        # vmap over faults...
        map_lnSA = jax.vmap(self.gmmlt.calc_all, in_axes = (None, None, None, 0, 0))
        # and over magnitudes
        map_lnSA = jax.vmap(map_lnSA, in_axes = (0, None, None, None, None))
        # Shape (number M bins, number faults, number gmms) [ijk]
        R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
        all_mu_lnSA, all_std_lnSA = map_lnSA(roots_M_fast, T, site, fault_tree, R_tree)
        # Shape (number ims, number M bins, number faults, number gmms) [hijk]
        all_exc_im = jax.vmap(self._calc_exc_im, in_axes = (0, None, None))(im, all_mu_lnSA, all_std_lnSA)

        # Take mean across gmms
        mu_exc_im = jnp.einsum('k,hijk->hij', self.gmmlt.w, all_exc_im)
        mu_haz = jnp.einsum('ijk,jk->i', mu_exc_im, n_M_incr_fast)
        return mu_haz
    
    def calc_epi_haz(self, im:jax.Array, T:float, site:Site, faults:list[Fault], M_ranges:jax.Array|None = None):
        # Get rates, bins, roots
        fault_tree = _faults_to_tree(faults)
        roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
        roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
        # vmap over faults...
        map_lnSA = jax.vmap(self.gmmlt.calc_all, in_axes = (None, None, None, 0, 0))
        # and over magnitudes
        map_lnSA = jax.vmap(map_lnSA, in_axes = (0, None, None, None, None))
        # Shape (number M bins, number faults, number gmms) [ijk]
        R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
        all_mu_lnSA, all_std_lnSA = map_lnSA(roots_M_fast, T, site, fault_tree, R_tree)
        # Shape (number ims, number M bins, number faults, number gmms) [hijk]
        all_exc_im = jax.vmap(self._calc_exc_im, in_axes = (0, None, None))(im, all_mu_lnSA, all_std_lnSA)

        # Take mean across gmms
        all_haz = jnp.einsum('ijkh,jk->ih', all_exc_im, n_M_incr_fast)
        mu_haz = jnp.einsum('ih,h->i', all_haz, self.gmmlt.w)
        std_haz = jnp.sqrt((all_haz - mu_haz[:, None])**2 @ self.gmmlt.w)
        return std_haz
    
    #### DISAGGREGATED HAZARD ####
    def calc_disagg(self, im:jax.Array, T:float, site:Site, faults:list[Fault], 
                    num_M_disagg:int = 8, num_R_disagg:int = 12,
                    M_ranges:jax.Array|None = None):
        # Get rates, bins, roots
        fault_tree = _faults_to_tree(faults)
        roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
        roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
        # vmap over faults...
        map_lnSA = jax.vmap(self.gmmlt.calc_all, in_axes = (None, None, None, 0, 0))
        # and over magnitudes
        map_lnSA = jax.vmap(map_lnSA, in_axes = (0, None, None, None, None))
        # Shape (number M bins, number faults, number gmms) [ijk]
        R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
        all_mu_lnSA, all_std_lnSA = map_lnSA(roots_M_fast, T, site, fault_tree, R_tree)
        # Shape (number ims, number M bins, number faults, number gmms) [hijk]
        all_exc_im = jax.vmap(self._calc_exc_im, in_axes = (0, None, None))(im, all_mu_lnSA, all_std_lnSA)

        # Take mean across gmms, Shape(number ims, number M bins, number faults)
        mu_exc_im = jnp.einsum('k,hijk->hij', self.gmmlt.w, all_exc_im)
        # Shape (number ims, number M bins, number faults)
        mu_disagg = mu_exc_im * n_M_incr_fast[None]

        # Project onto unmasked M bins (number ims, number full M bins, number faults)
        mu_disagg_full = jnp.zeros((im.shape[0], *n_M_incr.shape))
        mu_disagg_full = mu_disagg_full.at[:, M_mask].set(mu_disagg)

        # Aggregate into true M bins (number ims, number disagg M bins, number faults)
        disagg_M_bins = jnp.linspace(roots_M.min(), roots_M.max(), num_M_disagg)
        M_idcs = jnp.digitize(roots_M, disagg_M_bins)
        mu_disagg_M = jax.vmap(jax.vmap(jax.ops.segment_sum, in_axes = (0, None)), in_axes = (2, None), out_axes = 2)(mu_disagg_full, M_idcs)

        # Aggregate into R bins according to R tree (number ims, number disagg M bins, number disagg R bins)
        disagg_R_bins = jnp.linspace(R_tree[0].min(), R_tree[0].max(), num_R_disagg)
        R_idcs = jnp.digitize(R_tree[0], disagg_R_bins)
        mu_disagg_MR = jax.vmap(jax.vmap(jax.ops.segment_sum, in_axes = (0, None)), in_axes = (1, None), out_axes = 1)(mu_disagg_M, R_idcs)

        return mu_disagg_MR / mu_disagg_MR.sum(), disagg_M_bins, disagg_R_bins
    
    def _KLE(self, scn_vec:jax.Array, 
                params_idcs:jax.Array, params_transforms:list[Callable], 
                params_a:jax.Array, params_b:jax.Array, 
                n_sample:int, C:Callable,
                k_KL:int, m_os_KL:int, k_cheb_KL:int, k_phs_KL:int, k_poly_KL:int,
                key:jax.Array = jrnd.key(0)):
        # Split keys for sampling and randomized eigendecomposition
        key_params, key_rBK, = jrnd.split(key, num = 2)

        # Enumerate input parameters
        n_params = len(params_idcs)
        params_enum = pts_rqmc(params_a, params_b, n_sample, n_params, key_params).squeeze()
        # Transform to (better) space
        params_enum = jax.vmap(lax.switch, in_axes = (0, None, -1), out_axes = 1)(jnp.arange(n_params), params_transforms, params_enum)
        scn_enum = jnp.stack([scn_vec] * n_sample, axis = 0).at[:, params_idcs].set(params_enum)
        M_enum, T_enum, site_enum, fault_enum = jax.vmap(Scenario.objs_fromvec)(scn_enum)
        R_enum = jax.vmap(calc_R)(site_enum, fault_enum)

        # Get gmm outputs
        gmm_enum_mu, _ = jax.vmap(self.gmmlt.calc_all)(M_enum, T_enum, site_enum, fault_enum, R_enum)

        # Produce covariance kernel
        params_enum_scaled = 2 * (params_enum - params_a) / (params_b - params_a) - 1
        params_enum_scaled = params_enum_scaled * gmm_enum_mu.std(axis = -1).max()
        K = C(params_enum_scaled, gmm_enum_mu)
        # Quadrature weights
        w = jnp.full((n_sample,), 1 / n_sample)
        # Take eigendecomposition
        eigvals, eigvecs = rBK(K * w, k_KL, m_os_KL, k_cheb_KL, key = key_rBK)
        eigvals, eigvecs = eigvals.real, eigvecs.real

        scaled_eigfns = rbf_interpolator(params_enum, jnp.sqrt(eigvals)[None] * eigvecs, k_phs_KL, k_poly_KL)
        return scaled_eigfns

    #### KL-PCE UQ ####
    def KLPCE(self, scn_vec:jax.Array, 
                params_idcs:jax.Array, params_transforms:list[Callable], 
                params_a:jax.Array, params_b:jax.Array, 
                n_sample:int, C:Callable, 
                k_KL:int, m_os_KL:int, k_cheb_KL:int, k_phs_KL:int, k_poly_KL:int,
                key:jax.Array = jrnd.key(0)):
        # Split key
        key_KLE, key_PCE = jrnd.split(key, 2)

        # Grab eigenfunctions + eigenvalues
        scaled_eigfns = self._KLE(scn_vec, 
                                  params_idcs, params_transforms, params_a, params_b, 
                                  n_sample, C,
                                  k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL,
                                  key_KLE)

        def build_PCE(T:float, site:Site, faults:list[Fault], 
                      im:float,
                    m_sample:int, k_PCE:int, strategy:str = 'hyperbolic', q:float = 0.625,
                    order_matters:bool = False,
                    M_ranges:jax.Array|None = None,
                    key_PCE:jax.Array = key_PCE):
            ### THIS WILL BE A BROKEN-UP HAZARD CALCULATON ###
            fault_tree = _faults_to_tree(faults)
            roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
            # Just to speed up marginal calculations like the one we perform for the test scenario
            roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
            # vmap over faults...
            map_mean_lnSA = jax.vmap(self.gmmlt.calc_mean, in_axes = (None, None, None, 0, 0))
            # and over magnitudes
            map_mean_lnSA = jax.vmap(map_mean_lnSA, in_axes = (0, None, None, None, None))
            # Shape (number M bins, number faults)
            R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
            mean_mu_lnSA, mean_std_lnSA = map_mean_lnSA(roots_M_fast, T, site, fault_tree, R_tree)

            # Build input scenario vector using double vmap
            #   Shape (number M bins, number faults, k_KL)
            map_scn = jax.vmap(Scenario, in_axes = (0, None, None, None))
            map_scn = jax.vmap(map_scn, in_axes = (None, None, None, 0))
            eval_scn_vec = map_scn(roots_M_fast, T, site, fault_tree).tree_tovec().T
            # Extract partially correlated parameters
            eval_scn_params = eval_scn_vec[:, :, params_idcs]

            # Evaluate eigenfunctons at correlation parameters
            #   Shape (number M bins, k_KL)
            eval_scn_eigfns = jax.vmap(scaled_eigfns)(eval_scn_params)

            # Define wrapped hazard calculation at z
            def loghaz_wrapped_z(x):
                # Apply KL-expanded covariance function evaluated at
                #   input random variable
                mean_mu_hat_lnSA = mean_mu_lnSA + eval_scn_eigfns @ x

                # Finish hazard calculation
                exc_im = self._calc_exc_im(im, mean_mu_hat_lnSA, mean_std_lnSA)
                haz = jnp.einsum('ij,ij->', exc_im, n_M_incr_fast)
                return jnp.log(haz)
        
            # Take realizations of k_KL standard normal Gaussian random variables
            X = jrnd.normal(key_PCE, (m_sample, k_KL))

            # Produce hazard PCE
            loghaz_PCE = PCE(X, loghaz_wrapped_z, 
                        k_PCE = k_PCE, strategy = strategy, q = q, order_matters = order_matters)
            
            def haz_PCE(xi):
                return jnp.exp(loghaz_PCE(xi))
            
            return haz_PCE
        return build_PCE

    def KLtPCE(self, scn_vec:jax.Array, 
                params_idcs:jax.Array, params_transforms:list[Callable], 
                params_a:jax.Array, params_b:jax.Array, 
                n_sample:int, C:Callable, 
                k_KL:int, m_os_KL:int, k_cheb_KL:int, k_phs_KL:int, k_poly_KL:int,
                key:jax.Array = jrnd.key(0)):
        # Split key
        key_KLE, key_tPCE = jrnd.split(key, 2)

        # Grab eigenfunctions + eigenvalues
        scaled_eigfns = self._KLE(scn_vec, 
                                  params_idcs, params_transforms, params_a, params_b, 
                                  n_sample, C,
                                  k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL,
                                  key_KLE)

        def build_tPCE(T:float, site:Site, faults:list[Fault], 
                      im0:float, k_taylor:int,
                    m_sample:int, k_PCE:int, strategy:str = 'hyperbolic', q:float = 0.625,
                    order_matters:bool = False,
                    M_ranges:jax.Array|None = None,
                    key_PCE:jax.Array = key_tPCE):
            ### THIS WILL BE A BROKEN-UP HAZARD CALCULATON ###
            fault_tree = _faults_to_tree(faults)
            roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
            roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
            # vmap over faults...
            map_mean_lnSA = jax.vmap(self.gmmlt.calc_mean, in_axes = (None, None, None, 0, 0))
            # and over magnitudes
            map_mean_lnSA = jax.vmap(map_mean_lnSA, in_axes = (0, None, None, None, None))
            # Shape (number M bins, number faults)
            R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
            mean_mu_lnSA, mean_std_lnSA = map_mean_lnSA(roots_M_fast, T, site, fault_tree, R_tree)

            # Build input scenario vector using double vmap
            #   Shape (number M bins, number faults, k_KL)
            map_scn = jax.vmap(Scenario, in_axes = (0, None, None, None))
            map_scn = jax.vmap(map_scn, in_axes = (None, None, None, 0))
            eval_scn_vec = map_scn(roots_M_fast, T, site, fault_tree).tree_tovec().T
            # Extract partially correlated parameters
            eval_scn_params = eval_scn_vec[:, :, params_idcs]

            # Evaluate eigenfunctons at correlation parameters
            #   Shape (number M bins, k_KL)
            eval_scn_eigfns = jax.vmap(scaled_eigfns)(eval_scn_params)

            # Define wrapped hazard calculation at z
            def loghaz_wrapped(log_im, x, mean_mu_lnSA, mean_std_lnSA, eval_scn_eigscaled, n_M_incr_fast):
                # Apply KL-expanded covariance function evaluated at
                #   input random variable
                im = jnp.exp(log_im)
                mean_mu_hat_lnSA = mean_mu_lnSA + eval_scn_eigscaled @ x
                # Finish hazard calculation
                exc_im = self._calc_exc_im(im, mean_mu_hat_lnSA, mean_std_lnSA)
                haz = jnp.einsum('ij,ij->', exc_im, n_M_incr_fast)
                return jnp.log(haz)
            # Group other arguments
            haz_args = [mean_mu_lnSA, mean_std_lnSA, eval_scn_eigfns, n_M_incr_fast]
        
            # Take realizations of k_KL standard normal Gaussian random variables
            X = jrnd.normal(key_PCE, (m_sample, k_KL))

            # Produce hazard PCE
            loghaz_tPCE_logz = tPCE(X, loghaz_wrapped, jnp.log(im0), 
                            *haz_args,
                            k_taylor = k_taylor, k_PCE = k_PCE, 
                            strategy = strategy, q = q, order_matters = order_matters)
            def haz_tPCE(im_i, xi):
                return jnp.exp(loghaz_tPCE_logz(jnp.log(im_i), xi))
            
            return haz_tPCE
        return build_tPCE
    def tree_flatten(self):
        return (self.gmmlt, self.dM), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)