# Jaxlib
import jax
from jax import lax
from jax import random as jrnd
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.scipy.stats import norm

from numerics import *
from seismic import *

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
    # Magnitude/rate bins for integration
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
        M_mask = jnp.any(weights_mask, axis = -1)
        return roots_M, n_M_incr, M_mask
    
    # Exceedance from GMM outputs
    def _calc_all_exc_im(self, im:float, mu_lnSA:jax.Array, std_lnSA:jax.Array):
        # Calculate cumulative probability
        cum_prob_im = norm.cdf(jnp.log(im), loc = mu_lnSA, scale = std_lnSA)
        # Note that we SHOULD implement a 3-sigma upper bound here, but it appears to be
        #   incompatible with jax.jet...
        return 1 - cum_prob_im
    
    def _calc_pre_exc_im(self, ims:jax.Array, T:float, roots_M:jax.Array, 
                          site:Site, fault_tree:Fault, R_tree:jax.Array):
        # Fault map
        map_lnSA = jax.vmap(self.gmmlt.calc_median, in_axes = (None, None, None, 0, 0))
        # Magnitude map
        map_lnSA = jax.vmap(map_lnSA, in_axes = (0, None, None, None, None))
        # Get exceedance
        median_mu_lnSA, median_std_lnSA = map_lnSA(roots_M, T, site, fault_tree, R_tree)
        # Get average exceedance for all ims
        median_exc_im = jax.vmap(self._calc_all_exc_im, in_axes = (0, None, None))(ims, median_mu_lnSA, median_std_lnSA)
        return median_exc_im
    
    def _calc_post_exc_im(self, ims:jax.Array, T:float, roots_M:jax.Array, 
                          site:Site, fault_tree:Fault, R_tree:jax.Array):
        # Fault map
        map_lnSA = jax.vmap(self.gmmlt.calc_all, in_axes = (None, None, None, 0, 0))
        # Magnitude map
        map_lnSA = jax.vmap(map_lnSA, in_axes = (0, None, None, None, None))
        # Evaluate GMM
        all_mu_lnSA, all_std_lnSA = map_lnSA(roots_M, T, site, fault_tree, R_tree)
        # Get exceedances
        all_exc_im = jax.vmap(self._calc_all_exc_im, in_axes = (0, None, None))(ims, all_mu_lnSA, all_std_lnSA)
        # Average
        mu_exc_im = jnp.einsum('k,hijk->hij', self.gmmlt.w, all_exc_im)
        return mu_exc_im
    
    def _preavg_sigma(self, all_mu_lnSA:jax.Array, all_std_lnSA:jax.Array):
        median_mu_lnSA = all_mu_lnSA @ self.gmmlt.w
        median_std_lnSA = ((all_mu_lnSA**2 + all_std_lnSA**2) @ self.gmmlt.w - median_mu_lnSA ** 2) ** (1 / 2)
        return jnp.broadcast_to(median_std_lnSA[:, :, None], all_std_lnSA.shape)
    
    def _postavg_sigma(self, all_mu_lnSA:jax.Array, all_std_lnSA:jax.Array):
        return all_std_lnSA

    #### HAZARD CALC ####
    def calc_haz(self, ims:jax.Array, T:float, site:Site, faults:list[Fault], 
                 preavg:bool=False, 
                 M_ranges:jax.Array|None = None):
        # Fault tree
        fault_tree = _faults_to_tree(faults)
        # Get R-tree for faults
        R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
        # Get magnitude/rate roots + mask for marginal calc
        roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
        roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
        exc_im = lax.cond(preavg, self._calc_pre_exc_im, 
                                  self._calc_post_exc_im, 
                                  ims, T, roots_M_fast, site, fault_tree, R_tree)
        haz = jnp.einsum('ijk,jk->i', exc_im, n_M_incr_fast)
        return haz
    
    #### ENUMERATED EPISTEMIC UNCERTAINTY ####
    def calc_epi_haz(self, im:jax.Array, T:float, site:Site, faults:list[Fault], 
                     preavg:bool=False,
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
        all_std_lnSA = lax.cond(preavg, self._preavg_sigma, self._postavg_sigma, all_mu_lnSA, all_std_lnSA)
        # Shape (number ims, number M bins, number faults, number gmms) [hijk]
        all_exc_im = jax.vmap(self._calc_all_exc_im, in_axes = (0, None, None))(im, all_mu_lnSA, all_std_lnSA)
        # Take mean across gmms
        all_haz = jnp.einsum('hijk,ij->hk', all_exc_im, n_M_incr_fast)
        lnhaz = jnp.log(all_haz)
        mu_lnhaz = lnhaz @ self.gmmlt.w
        std_lnhaz = jnp.sqrt((lnhaz - mu_lnhaz[:, None])**2 @ self.gmmlt.w)
        std_haz = jnp.exp(std_lnhaz)
        return std_haz
    
    #### KL HELPER ####
    def KLE(self, scn_vec:jax.Array, 
                params_idcs:jax.Array, params_transforms:list[Callable], 
                params_a:jax.Array, params_b:jax.Array, 
                n_KL_samples:int, C:Callable,
                k_KL:int, m_os_KL:int, k_cheb_KL:int, k_phs_KL:int, k_poly_KL:int,
                key:jax.Array = jrnd.key(0)):
        # Split keys for sampling and randomized eigendecomposition
        key_params, key_rBK, = jrnd.split(key, num = 2)

        # Enumerate input parameters
        n_params = len(params_idcs)
        params_enum = pts_rqmc(params_a, params_b, n_KL_samples, n_params, key_params)
        # Transform to (better) space
        params_transf_enum = jax.vmap(lax.switch, in_axes = (0, None, -1), out_axes = 1)(jnp.arange(n_params), params_transforms, params_enum)
        params_transf_a = jax.vmap(lax.switch, in_axes = (0, None, 0))(jnp.arange(n_params), params_transforms, params_a)
        params_transf_b = jax.vmap(lax.switch, in_axes = (0, None, 0))(jnp.arange(n_params), params_transforms, params_b)
        scn_enum = jnp.stack([scn_vec] * n_KL_samples, axis = 0).at[:, params_idcs].set(params_transf_enum)
        M_enum, T_enum, site_enum, fault_enum = jax.vmap(Scenario.objs_fromvec)(scn_enum)
        R_enum = jax.vmap(calc_R)(site_enum, fault_enum)

        # Get gmm outputs
        all_mu_lnSA_enum, _ = jax.vmap(self.gmmlt.calc_all)(M_enum, T_enum, site_enum, fault_enum, R_enum)

        # Produce covariance kernel
        K = C(params_enum, all_mu_lnSA_enum, self.gmmlt.w)

        # Quadrature weights
        w_sqrt = jnp.sqrt(jnp.prod(params_transf_b - params_transf_a) / n_KL_samples)
        W_sqrt = jnp.full((1, n_KL_samples), w_sqrt)
        # Take eigendecomposition (same as diags on both sides but no materialization)
        eigvals, eigvecs = rBK(W_sqrt.T * K * W_sqrt, k_KL, m_os_KL, k_cheb_KL, key = key_rBK)
        eigvals, eigvecs = eigvals.real, eigvecs.real / w_sqrt

        # Scale transformed parameters for stable interpolation
        params_transf_enum_scaled = (params_transf_enum - params_transf_a) / (params_transf_b - params_transf_a)
        _rbf_eigfns = rbf_interpolator(params_transf_enum_scaled, eigvecs, k_phs_KL, k_poly_KL)

        def eigfns(params_transf_eval):
            params_transf_eval_scaled = (params_transf_eval - params_transf_a) / (params_transf_b - params_transf_a)
            return _rbf_eigfns(params_transf_eval_scaled)

        return eigvals, eigfns
    
    def _pce_haz_fit(self, x:jax.Array, im:float,
                         eval_scn_eigfns:jax.Array,
                         median_mu_lnSA:jax.Array, median_std_lnSA:jax.Array, 
                         n_M_incr:jax.Array,
                         dlnhaz:bool = False):
        # Apply KL-expanded covariance function evaluated at
        #   input random variable
        median_mu_hat_lnSA = median_mu_lnSA + eval_scn_eigfns @ x
        # Calculate exceedance
        exc_im = self._calc_all_exc_im(im, median_mu_lnSA, median_std_lnSA)
        exc_im_hat = self._calc_all_exc_im(im, median_mu_hat_lnSA, median_std_lnSA)
        # Finish (the quadrature-sum would distribute over the difference)
        haz = jnp.einsum('ij,ij->', exc_im, n_M_incr)
        haz_hat = jnp.einsum('ij,ij->', exc_im_hat, n_M_incr)
        # Update based on what we're fitting to...
        ret = jnp.where(dlnhaz, jnp.log(haz_hat / haz), haz_hat)
        return ret
    
    def _lnhaz_from_pce_haz(self, pce_out:jax.Array, median_haz):
        return jnp.log(jnp.clip(1e-8, pce_out))
    
    def _lnhaz_from_pce_dlnhaz(self, pce_out:jax.Array, median_haz):
        all_haz = jnp.log(median_haz)[:, None] + pce_out
        return all_haz

    def _pce_haz_eval(self, n_eval:int, key:jax.Array,
                      dlnhaz:bool, haz_pce:PCE, c:jax.Array,
                      median_haz:jax.Array, k_KL:int):
        xi = jrnd.normal(key, (n_eval, k_KL))
        pce_out = jax.vmap(haz_pce.eval_coeffs, in_axes = (None, 0))(xi, c)
        all_lnhaz = lax.cond(dlnhaz, self._lnhaz_from_pce_dlnhaz, 
                                     self._lnhaz_from_pce_haz,
                                     pce_out, median_haz)
        return jnp.exp(all_lnhaz.mean(axis = -1)), jnp.exp(jnp.quantile(all_lnhaz, 0.5, axis = -1)), jnp.exp(all_lnhaz.std(axis = -1)), jnp.exp(all_lnhaz)

    #### KL-PCE UQ ####
    def KLPCE(self, scn_vec:jax.Array, 
                params_idcs:jax.Array, params_transforms:list[Callable], 
                params_a:jax.Array, params_b:jax.Array, 
                n_KL_samples:int, C:Callable, 
                k_KL:int, m_os_KL:int, k_cheb_KL:int, k_phs_KL:int, k_poly_KL:int,
                key:jax.Array = jrnd.key(0)):
        # Split key
        keyKLE, key_PCE = jrnd.split(key, 2)

        # Grab eigenfunctions + eigenvalues
        eigvals, eigfns = self.KLE(scn_vec, 
                                  params_idcs, params_transforms, params_a, params_b, 
                                  n_KL_samples, C,
                                  k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL,
                                  keyKLE)

        def build_PCE(T:float, site:Site, faults:list[Fault], ims:jax.Array,
                    n_PCE_samples:int, k_PCE:int, strategy:str = 'hyperbolic', q:float = 0.625,
                    dlnhaz:bool = True,
                    M_ranges:jax.Array|None = None,
                    key:jax.Array = key_PCE):
            key_calc, key_eval = jrnd.split(key, 2)
            ### THIS WILL BE A BROKEN-UP HAZARD CALCULATON ###
            fault_tree = _faults_to_tree(faults)
            roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
            # Just to speed up marginal calculations like the one we perform for the test scenario
            roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
            # vmap over faults...
            map_mean_lnSA = jax.vmap(self.gmmlt.calc_median, in_axes = (None, None, None, 0, 0))
            # and over magnitudes
            map_mean_lnSA = jax.vmap(map_mean_lnSA, in_axes = (0, None, None, None, None))
            # Shape (number M bins, number faults)
            R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
            median_mu_lnSA, median_std_lnSA = map_mean_lnSA(roots_M_fast, T, site, fault_tree, R_tree)

            # Build input scenario vector using double vmap
            #   Shape (number M bins, number faults, k_KL)
            map_scn = jax.vmap(Scenario, in_axes = (0, None, None, None))
            map_scn = jax.vmap(map_scn, in_axes = (None, None, None, 0))
            eval_scn_vec = map_scn(roots_M_fast, T, site, fault_tree).tree_tovec().T
            # Extract partially correlated parameters
            eval_scn_params = eval_scn_vec[:, :, params_idcs]

            # Evaluate eigenfunctons at correlation parameters
            #   Shape (number M bins, number faults, k_KL)
            eval_scn_eigfns = jax.vmap(eigfns)(eval_scn_params)
            # Eigenvalue scaling...
            eval_scn_eigfns = eval_scn_eigfns * jnp.sqrt(eigvals[None, None, :])

            # Fit to the residual hazard between the mean and a perturbation
            pce_haz_args = [eval_scn_eigfns, median_mu_lnSA, median_std_lnSA, n_M_incr_fast, dlnhaz]

            # Take realizations of k_KL standard normal Gaussian random variables
            # But use RQMC. 
            X = jrnd.normal(key_calc, (n_PCE_samples, k_KL))
            #X = jax.vmap(norm.ppf)(pts_rqmc(0, 1, n_PCE_samples, k_KL, key = key_calc))
            # Fit PCE
            haz_pce = PCE(self._pce_haz_fit, k_PCE = k_PCE, 
                      strategy = strategy, q = q)
            c = jax.vmap(haz_pce.calc_coeffs, in_axes = (None, 0) + (None,) * len(pce_haz_args))(X, ims, *pce_haz_args)

            # Best way to do this at the moment...
            median_haz = self.calc_haz(ims, T, site, faults, True, M_ranges)

            def evaluate(C:jax.Array, median_haz:jax.Array, n_eval:int, key:jax.Array = key_eval):
                mu_haz, p50_haz, epi_haz, all_haz = self._pce_haz_eval(n_eval, key, dlnhaz, 
                                               haz_pce, C, median_haz, k_KL)
                return mu_haz, p50_haz, epi_haz, all_haz
                
            return c,median_haz,evaluate
        
        return build_PCE
    
    def KLtPCE(self, scn_vec:jax.Array, 
                params_idcs:jax.Array, params_transforms:list[Callable], 
                params_a:jax.Array, params_b:jax.Array, 
                n_KL_samples:int, C:Callable, 
                k_KL:int, m_os_KL:int, k_cheb_KL:int, k_phs_KL:int, k_poly_KL:int,
                key:jax.Array = jrnd.key(0)):
        print('Broken...')
        return 0
        # # Split key
        # keyKLE, key_tPCE = jrnd.split(key, 2)

        # # Grab eigenfunctions + eigenvalues
        # eigvals, eigfns = self.KLE(scn_vec, 
        #                           params_idcs, params_transforms, params_a, params_b, 
        #                           n_KL_samples, C,
        #                           k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL,
        #                           keyKLE)

        # def build_tPCE(T:float, site:Site, faults:list[Fault], IM0:jax.Array,
        #             n_PCE_samples:int, k_taylor:int, k_PCE:int, 
        #             strategy:str = 'hyperbolic', q:float = 0.625,
        #             dlnhaz:bool = True,
        #             M_ranges:jax.Array|None = None,
        #             key:jax.Array = key_tPCE):
        #     key_calc, key_eval = jrnd.split(key, 2)
        #     ### THIS WILL BE A BROKEN-UP HAZARD CALCULATON ###
        #     fault_tree = _faults_to_tree(faults)
        #     roots_M, n_M_incr, M_mask = self._calc_M_n_M_bins(fault_tree, M_ranges)
        #     # Just to speed up marginal calculations like the one we perform for the test scenario
        #     roots_M_fast, n_M_incr_fast = roots_M[M_mask], n_M_incr[M_mask]
        #     # vmap over faults...
        #     map_median_lnSA = jax.vmap(self.gmmlt.calc_median, in_axes = (None, None, None, 0, 0))
        #     # and over magnitudes
        #     map_median_lnSA = jax.vmap(map_median_lnSA, in_axes = (0, None, None, None, None))
        #     # Shape (number M bins, number faults)
        #     R_tree = jax.vmap(calc_R, in_axes = (None, 0))(site, fault_tree)
        #     median_mu_lnSA, median_std_lnSA = map_median_lnSA(roots_M_fast, T, site, fault_tree, R_tree)

        #     # Build input scenario vector using double vmap
        #     #   Shape (number M bins, number faults, k_KL)
        #     map_scn = jax.vmap(Scenario, in_axes = (0, None, None, None))
        #     map_scn = jax.vmap(map_scn, in_axes = (None, None, None, 0))
        #     eval_scn_vec = map_scn(roots_M_fast, T, site, fault_tree).tree_tovec().T
        #     # Extract partially correlated parameters
        #     eval_scn_params = eval_scn_vec[:, :, params_idcs]

        #     # Evaluate eigenfunctons at correlation parameters
        #     #   Shape (number M bins, number faults, k_KL)
        #     eval_scn_eigfns = jax.vmap(eigfns)(eval_scn_params)
        #     # Eigenvalue scaling...
        #     eval_scn_eigfns = eval_scn_eigfns * jnp.sqrt(eigvals[None, None, :])

        #     # Wrap up *args for _delta_lnhaz_wrapped
        #     delta_lnhaz_args = [eval_scn_eigfns, median_mu_lnSA, median_std_lnSA, n_M_incr_fast]

        #     # Define wrapped hazard calculation at z
        #     # "hatmul" is just a little helper to calculate median hazard further down because I am lazy
            
        #     def _haz_fit(ln_im0, x, eval_scn_eigfns:jax.Array,
        #                  median_mu_lnSA:jax.Array, median_std_lnSA:jax.Array, 
        #                  n_M_incr:jax.Array):
        #         return self._pce_haz_fit(x, jnp.exp(ln_im0), eval_scn_eigfns, median_mu_lnSA, median_std_lnSA, n_M_incr, dlnhaz)
        
        #     # Take realizations of k_KL standard normal Gaussian random variables
        #     X = jrnd.normal(key_calc, (n_PCE_samples, k_KL))

        #     for xi in X:
        #         plt.plot(ims, jax.vmap(_haz_fit, in_axes = (0, None) + (None,) * 4)(jnp.log(ims), xi, eval_scn_eigfns, median_mu_lnSA, median_std_lnSA, n_M_incr), lw = 0.8, alpha = 0.5)
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.show()

        #     # Fit PCE
        #     _lnhaz_tpce = tPCE(_haz_fit, k_taylor = k_taylor, k_PCE = k_PCE, 
        #               strategy = strategy, q = q)
             
        #     c_eval = _lnhaz_tpce.calc_coeffs(jnp.log(IM0), X, *delta_lnhaz_args)

        #     # Best way to do this at the moment...
        #     median_haz = self.calc_haz(ims, T, site, faults, True, M_ranges)

        #     def evaluate(ims:jax.Array, n_eval:int, key:jax.Array = key_eval):
        #         xi = jrnd.normal(key, (n_eval, k_KL))
        #         lnims = jnp.log(ims)
        #         all_dlnhaz = jax.vmap(_lnhaz_tpce.eval_coeffs, in_axes = (0, None, None))(lnims, xi, c_eval)
        #         return median_haz * jnp.exp(all_dlnhaz)
            
        #     return evaluate
        
        # return build_tPCE
     
    def tree_flatten(self):
        return (self.gmmlt, self.dM), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)