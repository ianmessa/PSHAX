# Jaxlib
import jax
jax.config.update('jax_enable_x64', True)
from jax import lax
from jax import random as jrnd
from jax import numpy as jnp
from jax import tree_util as jtu

# Others
from matplotlib import pyplot as plt
import os

# This
from numerics import *
from seismic import *
from hazcalc import *

# Simple site
x_site, y_site = 0., 0.
vs30 = 760
z1p0, z2p5 = 0.8, 0.
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
n_ims = 16
ims = jnp.logspace(-3, -1, n_ims, base = 10)
# Evaluation im
ime = ims[4]
T = 0.05

# Hazard calculator
dM = 0.04 # (just to speed things up compared to the default 0.01)
haz_calc = HazCalculator(gmmlt, dM)

### PHASE 1 ###
params_idcs = jnp.array([0, 8])
params_transforms = [lambda x: x, jnp.exp]
params_a = jnp.array([5., 1.6])
params_b = jnp.array([8., 5.2])
d = len(params_idcs)
n_KL_samples = 50

scn_fault1_vec = Scenario(0., T, site, fault1).tree_tovec()
scn_fault2_vec = Scenario(0., T, site, fault2).tree_tovec()

############################### PHASE 1:
# Eigenvalue comparison for both scenarios with all kernels
def phase1():
    for i,scn_vec in enumerate([scn_fault1_vec, scn_fault2_vec]):
        # Enumerate
        params_enum = pts_rqmc(params_a, params_b, n_KL_samples, d)
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
        gmm_mu_enum, _ = jax.vmap(haz_calc.gmmlt.calc_all)(M_enum, T_enum, site_enum, fault_enum, R_enum)
        if i == 0:
            jnp.savez('src/test_outputs/phase1/M_lnR_R_enum.npz', M_enum = M_enum, lnR_enum = lnR_enum, R_enum = jnp.exp(lnR_enum))
        print('Enumerated parameters saved')

        # Get inputs
        MlnR_enum = jnp.stack([M_enum, lnR_enum], axis = -1)
        MlnR_unit = 2 * (MlnR_enum - params_a) / (params_b - params_a) - 1
        MlnR_msp = MlnR_unit * gmm_mu_enum.std(axis = -1).max()
        MlnR_msq = MlnR_unit / gmm_mu_enum.std(axis = -1).max()
        MlnR_meansp = MlnR_unit * gmm_mu_enum.std(axis = -1).mean()

        K_Lacour = C_Lacour(MlnR_enum, gmm_mu_enum)
        K_Paciorek_unsc = C_Paciorek(MlnR_enum, gmm_mu_enum)
        K_Paciorek_unit = C_Paciorek(MlnR_unit, gmm_mu_enum)
        K_Paciorek_msp = C_Paciorek(MlnR_msp, gmm_mu_enum)
        K_Paciorek_msq = C_Paciorek(MlnR_msq, gmm_mu_enum)
        K_Paciorek_meansp = C_Paciorek(MlnR_meansp, gmm_mu_enum)

        Ks = jnp.stack([K_Lacour, K_Paciorek_unsc, K_Paciorek_unit, K_Paciorek_msp, K_Paciorek_msq, K_Paciorek_meansp], axis = 0)
        eigvals, eigvecs = jax.vmap(jnpla.eig)(Ks)

        names = ['Lacour', 'Paciorek_unscaled', 'Paciorek_unit', 'Paciorek_msp', 'Paciorek_msq', 'Paciorek_meansp']

        for K,eigval,eigvec,name in zip(Ks, eigvals, eigvecs, names):
            jnp.savez('src/test_outputs/phase1/scn%s_K_%s.npz'%(i+1,name), allow_pickle = True, K = K, eigval = eigval, eigvec = eigvec)
            print('saved scn%s, %s'%(i+1, name))
#phase1()

############################### PHASE 2: 
# Perfcomp with these specific hyperparameters
# Test scenario! (High vs30, near-field)
site_test1 = Site(x_site, y_site, 1080, 0.8, 0., 0.)
faults_test1 = [fault2]
# Test scenario! (Low vs30, deep basins, far-field)
site_test2 = Site(x_site, y_site, 260, 2.9, 0., 0.)
faults_test2 = [Fault(x_fault1, y_fault1, z_hyp, 18, 25., dip = 20, rake = 0., width = 3.1, HW_flag = 0, mfd = MFD(2.4, 1.3, 5.))]


n_KL_samples = 200
k_KL = 6
m_os_KL = 1.75
k_cheb_KL = 2
k_phs_KL = 3
k_poly_KL = 3

k_PCE = 5
k_taylor = 4
n_PCE_samples = 300
q = 0.625

n_eval = 750

def phase2(key, key_num, save_enum:bool = False):
    key1,key2,key3 = jrnd.split(key, 3)
    phase_dir = 'src/test_outputs/phase2'
    os.makedirs(phase_dir, exist_ok = True)
    for M_ranges,marg_name in zip([None, M_margs], ['full', 'marg']):
        print('----------%s----------'%marg_name.upper())
        marg_dir = os.path.join(phase_dir, marg_name)
        os.makedirs(marg_dir, exist_ok = True)
        if save_enum:
            median = haz_calc.calc_haz(ims, T, site, faults, M_ranges)
            epi84 = haz_calc.calc_epi_haz(ims, T, site, faults, M_ranges)
            enum_path = os.path.join(marg_dir, 'median_epi84_enum.npz')
            jnp.savez(enum_path, median_enum = median, epi84_enum = epi84)
            print('Enumeration saved')
        for scn_num_raw,scn_vec in enumerate([scn_fault1_vec, scn_fault2_vec]):
            build_pce = haz_calc.KLPCE(scn_vec, 
                                 params_idcs, params_transforms,
                                 params_a, params_b,
                                 n_KL_samples, C_Paciorek, 
                                 k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key1)
            # Update scenario vector. Almost forgot to do this.
            scn_num = scn_num_raw + 1
            # Initial scenario
            pce = build_pce(T, site, faults, ims, n_PCE_samples, k_PCE, M_ranges = M_ranges, key_PCE = key2)
            median_pce, epi84_pce = pce(n_eval, key3)
            pce_path = os.path.join(marg_dir, 'median_epi84_pce_scn%s_key%s.npz'%(scn_num, key_num))
            jnp.savez(pce_path, median_pce = median_pce, epi84_pce = epi84_pce)
            print("PCE %s saved"%scn_num)
            # Tests
            if marg_name == 'full':
                for test_num_raw,(site_test, faults_test) in enumerate([(site_test1, faults_test1), (site_test2, faults_test2)]):
                    test_num = test_num_raw + 1
                    pce = build_pce(T, site_test, faults_test, ims, n_PCE_samples, k_PCE, M_ranges = M_ranges, key_PCE = key2)
                    median_pce, epi84_pce = pce(n_eval, key3)
                    pce_test_path = os.path.join(marg_dir, 'median_epi84_pce_scn%s_test%s_key%s.npz'%(scn_num, test_num, key_num))
                    jnp.savez(pce_test_path, median_pce = median_pce, epi84_pce = epi84_pce)
                    print("PCE %s for test %s saved"%(scn_num, test_num))

            build_tpce = haz_calc.KLtPCE(scn_vec, 
                                 params_idcs, params_transforms,
                                 params_a, params_b,
                                 n_KL_samples, C_Paciorek, 
                                 k_KL, m_os_KL, k_cheb_KL, k_phs_KL, k_poly_KL, key1)
            
            tpce_path = os.path.join(marg_dir, 'median_epi84_tpce_scn%s_key%s.npz'%(scn_num, key_num))
            tpce_dict = {}
            for im0 in ims[2::4]:
                tpce = build_tpce(T, site, faults, im0, n_PCE_samples, k_taylor, k_PCE, M_ranges = M_ranges, key_PCE = key2)
                median_tpce, epi84_tpce = tpce(ims, n_eval, key3)
                tpce_dict['median_tpce_im0=%s'%f"{im0:.4e}".replace('.', 'p').replace('-','m')] = median_tpce
                tpce_dict['epi84_tpce_im0=%s'%f"{im0:.4e}".replace('.', 'p').replace('-','m')] = epi84_tpce
            jnp.savez(tpce_path, **tpce_dict)
            print('tPCE %s saved'%scn_num)

            if marg_name == 'full':
                for test_num_raw,(site_test, faults_test) in enumerate([(site_test1, faults_test1), (site_test2, faults_test2)]):
                    test_num = test_num_raw + 1
                    tpce_dict = {}
                    for im0 in ims[2::4]:
                        tpce = build_tpce(T, site_test, faults_test, im0, n_PCE_samples, k_taylor, k_PCE, M_ranges = M_ranges, key_PCE = key2)
                        median_tpce, epi84_tpce = tpce(ims, n_eval, key3)
                        tpce_dict['median_tpce_im0=%s'%f"{im0:.4e}".replace('.', 'p').replace('-','m')] = median_tpce
                        tpce_dict['epi84_tpce_im0=%s'%f"{im0:.4e}".replace('.', 'p').replace('-','m')] = epi84_tpce
                    tpce_test_path = os.path.join(marg_dir, 'median_epi84_tpce_scn%s_test%s_key%s.npz'%(scn_num, test_num, key_num))
                    jnp.savez(tpce_test_path, **tpce_dict)
                    print('tPCE %s for test %s saved'%(scn_num, test_num))

save_enum = False
for key_num,key in enumerate([jrnd.key(0), *jrnd.split(jrnd.key(5), 5)]):
   phase2(key, key_num, save_enum)
   save_enum = False

############################### PHASE 3:
# Speed profiling
# 