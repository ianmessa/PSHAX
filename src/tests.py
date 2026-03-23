# Jaxlib
import jax
jax.config.update('jax_enable_x64', True)
from jax import lax
from jax import random as jrnd
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.scipy.special import factorial

# Others
from matplotlib import pyplot as plt
import os

# This
from numerics import *
from seismic import *
from hazcalc import *

#### GLOBALS ####
key = jrnd.key(3)
# GMMs
gmms = [gmm_ASK14, gmm_BSSA14, gmm_CB14, gmm_CY14, gmm_Idriss14]
c_AAY14 = 1.674
gmms = gmms + [gmm_epi_AAY14(gmm, c_AAY14) for gmm in gmms] + [gmm_epi_AAY14(gmm, -c_AAY14) for gmm in gmms]
gmm_labels = ['ASK14', 'BSSA14', 'CB14', 'CY14', 'I14']

# GMM weights
w_gmms = jnp.ones(len(gmms)) / len(gmms)
gmmlt = GMMLT(gmms, w_gmms)

# Intensity measures + period
n_ims = 16
ims = jnp.logspace(-3, -1, n_ims, base = 10)
# Evaluation im
ime = ims[4]
T = 0.05

# Hazard calculator
dM = 0.01
haz_calc = HazCalculator(gmmlt, dM)

# Kernel tests
C_fns = [C_Lacour, C_Paciorek_iso, C_Paciorek_aniso, C_svarg_iso, C_svarg_aniso]
C_names = ['Lacour', 'PaciorekIso', 'PaciorekAniso', 'SvargIso', 'SvargAniso']

#### SCENARIO ####
# Simple site
x_site, y_site = 0., 0.
vs30_site1 = 760
vs30_site2 = 1500
z1p0_site1, z1p0_site2 = 0.8, 2.9
z2p5 = 0.
site1 = Site(x_site, y_site, 760, 0.8, 0., 0.)
site2 = Site(x_site, y_site, 1500, 0.8, 0., 0.)
site3 = Site(x_site, y_site, 1500, 2.9, 0., 0.)

# Fault 1 is large, faraway earthquakes; fault 2 is small, close
x_fault1, y_fault1 = 50., 49.
theta1, width1 = 225, 1.5
mfd1, M_marg1 = MFD(1.4, 1., M_min = 5.), jnp.array([7., 8.5])
x_fault2, y_fault2 = -20., 15.
theta2, width2 = 30, 2.7
mfd2, M_marg2 = MFD(2.6, 1.4, M_min = 5.), jnp.array([5., 6.])
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

params_idcs = jnp.array([0, 8])
params_transforms = [lambda x: x, jnp.exp]
params_a = jnp.array([5., 0.])
params_b = jnp.array([8., 5.298])
d = len(params_idcs)
n_KL_samples = 50

############################### PHASE 1:
# Eigenvalue comparison for both scenarios with all kernels
def phase1(key):
    os.makedirs('src/test_outputs/phase1', exist_ok = True)
    for i,fault in enumerate([fault1, fault2]):
        scn_vec = Scenario(0., T, site1, fault).tree_tovec()
        # Enumerate
        params_enum = pts_rqmc(params_a, params_b, n_KL_samples, d, key)
        # Save lnR
        lnR_enum = params_enum[:,-1]
        # Apply transformations
        params_enum = jax.vmap(lax.switch, in_axes = (0, None, -1), out_axes = 1)(jnp.arange(d), params_transforms, params_enum)
        # Enumerate scenario
        scn_enum = jnp.stack([scn_vec] * n_KL_samples, axis = 0).at[:, params_idcs].set(params_enum)
        # Unpack to objects
        M_enum, T_enum, site_enum, fault_enum = jax.vmap(Scenario.objs_fromvec)(scn_enum)
        R_enum = jax.vmap(calc_R)(site_enum, fault_enum)
        # Get gmm outputs
        gmm_median_enum, _ = jax.vmap(haz_calc.gmmlt.calc_all)(M_enum, T_enum, site_enum, fault_enum, R_enum)
        if i == 0:
            jnp.savez('src/test_outputs/phase1/enum.npz', M_enum = M_enum, lnR_enum = lnR_enum, R_enum = jnp.exp(lnR_enum), gmm_median_enum = gmm_median_enum)

        # Get inputs
        MlnR_enum = jnp.stack([M_enum, lnR_enum], axis = -1)

        for C,C_name in zip(C_fns, C_names):
            K = C(MlnR_enum, gmm_median_enum, w_gmms)

            eigval, eigvec = jnpla.eigh(K)
            jnp.savez('src/test_outputs/phase1/scn%s_K_%s.npz'%(i+1,C_name), allow_pickle = True, K = K, eigval = eigval, eigvec = eigvec)

############################### PHASE 1.5: 
# Efficacy of KL decomposition
m_os_KL = 1.75
k_cheb_KL = 3
k_phs_KL = 3
k_poly_KL = 2
n_sample = 250

def phase1p5(key):
    key0,key1 = jrnd.split(key, 2)
    phase_dir = 'src/test_outputs/phase1p5'
    roots_M = jnp.arange(5., 8.05, 0.05)
    os.makedirs(phase_dir, exist_ok = True)
    for i,fault in enumerate([fault1, fault2]):
        scn = Scenario(0., T, site1, fault)
        scn_vec = scn.tree_tovec()
        R = scn_vec[8]
        median_gmm, _ = jax.vmap(gmmlt.calc_median, in_axes = (0, None, None, None, None))(roots_M, T, site1, fault, calc_R(site1, fault))
        all_gmm, all_std_gmm = jax.vmap(gmmlt.calc_all, in_axes = (0, None, None, None, None))(roots_M, T, site1, fault, calc_R(site1, fault))
        epi84_gmm = all_gmm.std(axis = -1)
        jnp.savez(os.path.join(phase_dir, 'fault%s_lnSA_all_median_epi84'%(i+1)), all_gmm = all_gmm, median_gmm = median_gmm, epi84_gmm = epi84_gmm)
        for j,k_KL in enumerate([2, 3, 4, 5, 6, 12]):
            for C,C_name in zip(C_fns, C_names):
                eigvals, eigfns = haz_calc.KLE(scn_vec, params_idcs, params_transforms, 
                                params_a, params_b, 
                                n_KL_samples, C, 
                                k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key0)
                if jnp.any(eigvals < 0):
                    print('%s has a negative eigenvalue (position %s)'%(C_name, jnp.where(eigvals < 0)[0]))
                else:
                    eigfns_eval = eigfns(jnp.stack([roots_M, jnp.full_like(roots_M, R)], axis = -1)) * (eigvals[None, :]) ** (1 / 2)
                    X_test = jrnd.normal(key1, (n_sample, k_KL))
                    all_KL_gmm = median_gmm[:, None] + eigfns_eval @ X_test.T
                    median_KL_gmm, epi84_KL_gmm = all_KL_gmm.mean(axis = -1), all_KL_gmm.std(axis = -1)
                    jnp.savez(os.path.join(phase_dir, 'fault%s_KL%s_%s_lnSA_all_median_epi84'%(i+1, k_KL, C_name)), all_gmm = all_KL_gmm, median_gmm = median_KL_gmm, epi84_gmm = epi84_KL_gmm)
                    key1, _ = jrnd.split(key1, 2)

############################### PHASE 2:
# Uncertainty estimates for faults
n_eval = 5000
def phase2(key, C, C_name, k_KL:int):
    phase_dir = 'src/test_outputs/phase2'
    os.makedirs(phase_dir, exist_ok = True)
    pce_dir = os.path.join(phase_dir, 'pce')
    tpce_dir = os.path.join(phase_dir, 'tpce')
    for dir in [pce_dir, tpce_dir]:
        os.makedirs(dir, exist_ok = True)
    for i,fault in enumerate([fault1, fault2]):
        scn = Scenario(0., T, site1, fault)
        scn_vec = scn.tree_tovec()
        mean_haz = haz_calc.calc_preavg_haz(ims, T, site1, [fault])
        epi84_haz = haz_calc.calc_epi_haz(ims, T, site1, [fault])
        jnp.savez(os.path.join(phase_dir, 'fault%s_haz_mean_epi84'%(i+1)), mean_haz = mean_haz, epi84_haz = epi84_haz)

        build_pce = haz_calc.KLPCE(scn_vec, params_idcs, params_transforms, 
                            params_a, params_b, 
                            n_KL_samples, C, 
                            k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key)
        for k_PCE in [4, 5, 6]:
            for q in [0.25, 0.5, 0.75]:
                numerator = factorial(k_KL + k_PCE)
                denominator = factorial(k_KL) * factorial(k_PCE)
                n_PCE_samples = (numerator / denominator * (3 + 3 * q)).astype(int).item()
                built_pce = build_pce(T, site1, [fault], ims, n_PCE_samples, k_PCE, q = q)
                mean_haz_pce, epi84_haz_pce = built_pce(n_eval)
                jnp.savez(os.path.join(pce_dir, 'fault%s_%s_kpce%s_q%s_haz_mean_epi84'%(i+1, C_name, k_PCE, str(q).replace('.','p'))), 
                        mean_haz = mean_haz_pce, epi84_haz = epi84_haz_pce)

############################### PHASE 2.5:
# Cumulative scenario evaluation
def phase2p5(key, C, C_name, k_KL:int, k_PCE:int, q:float ):
    phase_dir = 'src/test_outputs/phase2p5'
    os.makedirs(phase_dir, exist_ok = True)
    pce_dir = os.path.join(phase_dir, 'pce')
    for dir in [pce_dir]:
        os.makedirs(dir, exist_ok = True)
    mean_haz = haz_calc.calc_preavg_haz(ims, T, site1, [fault1, fault2])
    epi84_haz = haz_calc.calc_epi_haz(ims, T, site1, [fault1, fault2])
    jnp.savez(os.path.join(phase_dir, 'fullscn_haz_mean_epi84'), mean_haz = mean_haz, epi84_haz = epi84_haz)
    for i,fault in enumerate([fault1, fault2]):
        scn = Scenario(0., T, site1, fault)
        scn_vec = scn.tree_tovec()
        
        build_pce = haz_calc.KLPCE(scn_vec, params_idcs, params_transforms, 
                            params_a, params_b, 
                            n_KL_samples, C, 
                            k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key)

        numerator = factorial(k_KL + k_PCE)
        denominator = factorial(k_KL) * factorial(k_PCE)
        n_PCE_samples = (numerator / denominator * (3 + 3 * q)).astype(int).item()
        built_pce = build_pce(T, site1, [fault1, fault2], ims, n_PCE_samples, k_PCE, q = q)
        mean_haz_pce, epi84_haz_pce = built_pce(n_eval)
        jnp.savez(os.path.join(pce_dir, 'fullscn_fitfault%s_%s_kpce%s_q%s_haz_mean_epi84'%(i+1, C_name, k_PCE, str(q).replace('.','p'))), 
                mean_haz = mean_haz_pce, epi84_haz = epi84_haz_pce)
        
############################### PHASE 2.75:
# Cumulative scenario evaluation with marginal
def phase2p75(key, C, C_name, k_KL:int, k_PCE:int, q:float ):
    phase_dir = 'src/test_outputs/phase2p75'
    os.makedirs(phase_dir, exist_ok = True)
    pce_dir = os.path.join(phase_dir, 'pce')
    for dir in [pce_dir]:
        os.makedirs(dir, exist_ok = True)
    mean_haz = haz_calc.calc_preavg_haz(ims, T, site1, [fault1, fault2], M_margs)
    epi84_haz = haz_calc.calc_epi_haz(ims, T, site1, [fault1, fault2], M_margs)
    jnp.savez(os.path.join(phase_dir, 'fullscn_marg_haz_mean_epi84'), mean_haz = mean_haz, epi84_haz = epi84_haz)
    for i,fault in enumerate([fault1, fault2]):
        scn = Scenario(0., T, site1, fault)
        scn_vec = scn.tree_tovec()
        
        build_pce = haz_calc.KLPCE(scn_vec, params_idcs, params_transforms, 
                            params_a, params_b, 
                            n_KL_samples, C, 
                            k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key)

        numerator = factorial(k_KL + k_PCE)
        denominator = factorial(k_KL) * factorial(k_PCE)
        n_PCE_samples = (numerator / denominator * (3 + 3 * q)).astype(int).item()
        built_pce = build_pce(T, site1, [fault1, fault2], ims, n_PCE_samples, k_PCE, q = q, M_ranges = M_margs)
        mean_haz_pce, epi84_haz_pce = built_pce(n_eval)
        jnp.savez(os.path.join(pce_dir, 'fullscn_marg_fitfault%s_%s_kpce%s_q%s_haz_mean_epi84'%(i+1, C_name, k_PCE, str(q).replace('.','p'))), 
                mean_haz = mean_haz_pce, epi84_haz = epi84_haz_pce)

            
############################### PHASE 3:
# Taylor method
def phase3(key, C, C_name, k_KL:int, k_PCE:int, q:float, fault_num:int):
    phase_dir = 'src/test_outputs/phase3'
    os.makedirs(phase_dir, exist_ok = True)
    tpce_dir = os.path.join(phase_dir, 'tpce')
    for dir in [tpce_dir]:
        os.makedirs(dir, exist_ok = True)
    fault = [fault1, fault2][fault_num - 1]

    scn = Scenario(0., T, site1, fault)
    scn_vec = scn.tree_tovec()

    numerator = factorial(k_KL + k_PCE)
    denominator = factorial(k_KL) * factorial(k_PCE)
    n_PCE_samples = (numerator / denominator * (3 + 3 * q)).astype(int).item()
    
    build_pce = haz_calc.KLtPCE(scn_vec, params_idcs, params_transforms, 
                        params_a, params_b, 
                        n_KL_samples, C, 
                        k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key)
        
    for k_taylor in [3, 5, 10, 20]:
        mean_haz_tpce = []
        epi84_haz_tpce = []
        for im0 in ims[1::4]:
            built_tpce = build_pce(T, site1, [fault1, fault2], im0, n_PCE_samples, k_taylor, k_PCE, q = q)
            mean_haz_tpce_im0, epi84_haz_tpce_im0 = built_tpce(ims, n_eval)
            mean_haz_tpce.append(mean_haz_tpce_im0)
            epi84_haz_tpce.append(epi84_haz_tpce_im0)
        savename = 'fullscn_fitfault%s_%s_kpce%s_q%s'%(fault_num, C_name, k_PCE, str(q).replace('.','p')) + '_tayl%s_haz_im0idx_mean_epi84'%(k_taylor)
        mean_haz_tpce, epi84_haz_tpce = jnp.array(mean_haz_tpce), jnp.array(epi84_haz_tpce)
        jnp.savez(os.path.join(tpce_dir, savename), 
            im0idx = ims[1::4], mean_haz = mean_haz_tpce, epi84_haz = epi84_haz_tpce)

############################### PHASE 4:
# Generalizing to high vs30. Testing calibration for:
# Just MlnR to show poor generalization
# MlnR + vs30 on Site1 (no basin depth), Site2 (deep basin) to show sensitivity
#   to excluded parameters
def phase4(key, C, C_name, k_KL:int, k_PCE:int, q:float):
    phase_dir = 'src/test_outputs/phase4'
    os.makedirs(phase_dir, exist_ok = True)
    all_params_idcs = [jnp.array([0, 8]), jnp.array([0, 4, 8])]
    all_params_transforms = [[lambda x: x, jnp.exp], [lambda x: x, lambda x: x, jnp.exp]]
    all_params_a = [jnp.array([5., 0.]), jnp.array([5., 500, 0.])]
    all_params_b = [jnp.array([8., 5.298]), jnp.array([8., 1500, 5.298])]
    scn_vec = Scenario(0., T, site1, fault1).tree_tovec()
    for i,site_test in enumerate([site2, site3]):
        mean_haz = haz_calc.calc_preavg_haz(ims, T, site_test, [fault1, fault2], M_margs)
        epi84_haz = haz_calc.calc_epi_haz(ims, T, site_test, [fault1, fault2], M_margs)
        jnp.savez(os.path.join(phase_dir, 'site%s_scn_haz_mean_epi84'%((i + 2))), 
                          mean_haz = mean_haz, epi84_haz = epi84_haz)
        # For the deep basin site we're not gonna do the MlnR vs MlnRvs30 test
        if i == 1:
            all_params_idcs = all_params_idcs[1:]
            all_params_transforms = all_params_transforms[1:]
            all_params_a, all_params_b = all_params_a[1:], all_params_b[1:]
        for j,(params_idcs, params_transforms, params_a, params_b) in enumerate(zip(all_params_idcs, 
                                                                        all_params_transforms, 
                                                                        all_params_a, 
                                                                        all_params_b)):
            if i == 0 and j == 0:
                vs30str = ''
                vs30factor = 1
            else: 
                vs30str = 'vs30inc_'
                vs30factor = 3
            build_pce = haz_calc.KLPCE(scn_vec, params_idcs, params_transforms, 
                        params_a, params_b, 
                        n_KL_samples * vs30factor, C, 
                        k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key)

            numerator = factorial(k_KL + k_PCE)
            denominator = factorial(k_KL) * factorial(k_PCE)
            n_PCE_samples = (numerator / denominator * (3 + 3 * q)).astype(int).item()

            built_pce = build_pce(T, site_test, [fault1, fault2], ims, n_PCE_samples, k_PCE, q = q, M_ranges = M_margs)
            mean_haz_pce, epi84_haz_pce = built_pce(n_eval)
            jnp.savez(os.path.join(phase_dir, '%ssite%s_scn_%s_pce_haz_mean_epi84'%(vs30str, (i + 2), C_name)), 
                        mean_haz = mean_haz_pce, epi84_haz = epi84_haz_pce)








# print('Phase 1')
# phase1(key)
# print('Phase 1.5')
# phase1p5(key)
# print('Phase 2')
# phase2(key, C_Paciorek_iso, 'PaciorekIso', 6)
# print('Phase 2.5')
# phase2p5(key, C_Paciorek_iso, 'PaciorekIso', 6, 6, 0.25)
# print('Phase 2.75')
# phase2p75(key, C_Paciorek_iso, 'PaciorekIso', 6, 6, 0.25)
# phase2p75(key, C_fullcor, 'FullCorr', 2, 6, 0.25)
# print('Phase 3')
# phase3(key, C_Paciorek_iso, 'PaciorekIso', 6, 6, 0.25, 1)
print('Phase 4')
# Tripled number of KL samples in phase4 definition to account for extra dim
phase4(key, C_Paciorek_iso, 'PaciorekIso', 6, 6, 0.25)
phase4(key, C_Paciorek_aniso, 'PaciorekAniso', 9, 4, 0.25)