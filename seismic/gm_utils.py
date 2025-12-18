import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import tree as jt
from jax.scipy.stats.truncnorm import cdf as trunc_norm_cdf

from numerics import *

T_master = jnp.array([0.01, 0.02, 0.03, 0.05, 
                      0.075, 0.1, 0.15, 0.2,
                      0.25, 0.3, 0.4, 0.5,
                      0.75, 1., 1.5, 2., 3.,
                      4., 5., 6., 7.5, 10.,
                      -1.])

# Calculate SOF from rake
def rake_SOF_flag(rake:float):
    """
    Calculate style of fault from rake.

    Args
    ----------
    rake : float
        Fault rake (degrees).

    Returns
    -------
    SOF flag : float
        [reverse, SS, normal] -> [-1., 0., 1.]
    """
    abs_rake = jnp.abs(rake)
    is_SS = (~jnp.logical_and(abs_rake > 30, abs_rake < 150))
    # [reverse, SS, normal] -> [-1., 0., 1.]
    return (is_SS * - jnp.sign(rake))

@jtu.register_pytree_node_class
class MFD:
    def __init__(self, a:float, b:float, 
                       M_min:float = 4.0, M_max:float = 8.0):
        """
        Magnitude Frequency Dist. represented by a Gutenberg–Richter magnitude
        frequency distribution.

        Args
        ----------
        a : float
            Log10 productivity (a-value) of the Gutenberg–Richter relation.
        b : float
            b-value (slope) of the Gutenberg–Richter relation.
        M_min : float, optional
            Minimum magnitude (default 4.0).
        M_max : float, optional
            Maximum magnitude (default 8.0).
        """
        self.a, self.b = a, b
        self.M_min, self.M_max = M_min, M_max

    def calc_lmdaM(self, M):
        """Calculate rate of earthquakes exceeding magnitude M."""
        return 10 ** (self.a - self.b * M)

    def calc_FM(self, M):
        """Calculate probability of magnitude exceedance."""
        return 1 - 10 ** -(self.b * (M - self.M_min))
    
    def calc_fM(self, M):
        """Calculate probability for single magnitude occurrence."""
        return (self.b * jnp.log(2.1) * (1 - self.calc_FM(M))) / self.calc_FM(self.M_max)
    
    def tree_flatten(self):
        return (self.a, self.b, self.M_min, self.M_max), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# Site...
@jtu.register_pytree_node_class
class Site:
    def __init__(self, x:float, y:float, 
                       vs30:float, z1p0:float, z2p5:float, vs30inf_flag:float):
        self.x, self.y = x, y

        self.vs30, self.z1p0, self.z2p5, = vs30, z1p0, z2p5
        self.vs30inf_flag = vs30inf_flag
    
    def calc_xy(self):
        return jnp.array([self.x, self.y]).T
    
    def calc_xyz(self):
        return jnp.array([self.x, self.y, 0]).T

    def tree_flatten(self):
        return (self.x, self.y, self.vs30, self.z1p0, self.z2p5, self.vs30inf_flag), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
    
# Fault...
@jtu.register_pytree_node_class
class Fault:
    def __init__(self, x:float, y:float, z_hyp:float, z_tor:float, theta:float,
                       dip:float, rake:float, width:float,
                       HW_flag:float, mfd:MFD):
        """
        Initialize a Fault object.

        Args
        ----------
        x, y : float
            Planar coordinates of the fault hypocenter (cartesian, km).
        z_hyp : float
            Depth (z) of the hypocenter (km).
        z_tor : float
            Depth (z) to the top end of the rupture (km).
        theta : float
            Fault strike orientation (degrees). Convention: theta = 0 means fault is
            east-west and down-dip is north.
        dip : float
            Fault dip angle (degrees).
        rake : float
            Fault rake angle (degrees).
        width : float
            Rupture width (extent down-dip) (km).
        HW_flag : float
            Hanging-wall indicator / half-width parameter. Float, but should be 0 or 1.
        mfd : MFD
            Earthquake Rupture Forecast object associated with fault.
        """
        
        self.x, self.y = x, y
        self.z_hyp, self.z_tor = z_hyp, z_tor

        self.theta = theta

        self.dip, self.rake, self.width = dip, rake, width

        self.HW_flag = HW_flag

        self.mfd = mfd
    
    def calc_xy_hyp(self):
        """
        Return hypocentral location on xy grid.
        """
        return jnp.array([self.x, self.y]).T
    
    def calc_xyz_hyp(self):
        """
        Calculate hypocentral location on xyz grid.
        """
        return jnp.concat([self.calc_xy_hyp(), jnp.array([self.z_hyp]).T], axis = -1)
    
    def calc_dxyz_tor(self):
        """ 
        Calculate distance between top of rupture and hypocenter on xyz grid.
        """
        xyz_hyp = self.calc_xyz_hyp()
        # z distance between hypocenter and TOR
        dz = self.z_tor - xyz_hyp[2]
        # xy change distance hyporcenter and TOR
        dr = jnp.tan(self.dip * jnp.pi / 180) * dz
        # x, y diffs (note the unintuitively flipped sin/cos;
        #   this is because the fault surface is orthogonal to the
        #   fault's east/west orientation described by theta)
        dx = - dr * jnp.sin(self.theta * jnp.pi / 180)
        dy = dr * jnp.cos(self.theta * jnp.pi / 180)
        return jnp.array([dx, dy, dz])
    
    def calc_SOF_flag(self):
        """"
        Calculate SOF flag ([reverse, SS, normal] -> [-1., 0., 1.]) from rake.
        """
        return rake_SOF_flag(self.rake)

    def tree_flatten(self):
        return (self.x, self.y, self.z_hyp, self.z_tor, self.theta, self.dip, self.rake, self.width, self.HW_flag, self.mfd), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

def make_fault_tree(*faults):
    return jt.map(lambda *xs: jnp.stack(xs), *faults)

# Distances...
# xy Distance from nearest edge of rupture
def calc_R_jb(site: Site, fault: Fault) -> float:
    """Calculate 2D Joyner–Boore distance (horizontal distance to surface projection of rupture)."""

    # Grab coordinates
    site_xy = site.calc_xy()
    fault_xy_hyp = fault.calc_xy_hyp()
    fault_dxy_tor = fault.calc_dxyz_tor()[:-1]  # just x,y components

    # Distance between hypocenter and top-of-rupture (TOR)
    fault_dr_tor = jnp.linalg.norm(fault_dxy_tor, ord=2)

    # Projected down-dip width (horizontal projection)
    width_proj = jnp.cos(fault.dip * jnp.pi / 180) * fault.width
    scaling = (width_proj - fault_dr_tor) / fault_dr_tor

    # Define fault segment in map view
    edge1_xy = fault_xy_hyp + fault_dxy_tor
    edge2_xy = fault_xy_hyp - fault_dxy_tor * scaling

    # Vectors from site to fault edges
    edge1_dxy = edge1_xy - site_xy
    edge2_dxy = edge2_xy - site_xy
    span_xy = edge2_xy - edge1_xy

    # Distances to edges
    edge1_r = jnp.linalg.norm(edge1_dxy, ord=2)
    edge2_r = jnp.linalg.norm(edge2_dxy, ord=2)

    # Compute perpendicular distance to the fault segment (2D analog of cross product)
    span_r = jnp.abs(jnp.cross(span_xy, edge1_dxy)) / jnp.linalg.norm(span_xy, ord=2)

    # Project site onto fault trace to see if it lies beyond the endpoints
    proj_coeff = (site_xy - edge1_xy) @ span_xy / (span_xy @ span_xy)
    proj_xy = edge1_xy + proj_coeff * span_xy

    # Condition for whether projection lies outside fault segment
    edge_cond = jnp.logical_or(proj_coeff < 0.0, proj_coeff > 1.0)

    # Distance to nearest edge or perpendicular projection
    edge_r = jnp.minimum(edge1_r, edge2_r)
    return lax.select(edge_cond, edge_r, span_r)

# xyz Distance from rupture surface
def calc_R_rup(site: Site, fault: Fault) -> float:
    """Calculate 3d distance between site and nearest part of rupture"""
    # Grab coordinates
    site_xyz = site.calc_xyz()
    fault_xyz_hyp = fault.calc_xyz_hyp()
    fault_dxyz_tor = fault.calc_dxyz_tor()
    # Overall distance between hypocenter and TOR
    fault_dr_tor = jnp.linalg.norm(fault_dxyz_tor, ord = 2)
    # Distance for down-dip edge (edge2)
    scaling = (fault.width - fault_dr_tor) / fault_dr_tor
    # Fault edges
    edge1_xyz = fault_xyz_hyp + fault_dxyz_tor
    edge2_xyz = fault_xyz_hyp - fault_dxyz_tor * scaling

    # Distance vectors
    edge1_dxyz = edge1_xyz - site_xyz
    edge2_dxyz = edge2_xyz - site_xyz
    span_xyz = edge2_xyz - edge1_xyz

    # Distances
    edge1_r = jnp.linalg.norm(edge1_dxyz, ord = 2)
    edge2_r = jnp.linalg.norm(edge2_dxyz, ord = 2)
    span_xyz_cross = jnp.linalg.cross(span_xyz, edge1_dxyz)
    span_r = jnp.linalg.norm(span_xyz_cross, ord = 2) / jnp.linalg.norm(span_xyz, ord = 2)
    proj_xyz = edge1_xyz + (edge1_dxyz @ span_xyz) / (span_xyz @ span_xyz) * span_xyz
    proj_z = proj_xyz[-1]

    # Condition for edge vs. other distance
    edge_r = jnp.minimum(edge1_r, edge2_r)
    edge_cond = jnp.logical_or(proj_z > edge2_xyz[-1], proj_z < fault.z_tor)
    return lax.select(edge_cond, edge_r, span_r)

# xy Distance from epicenter
def calc_R_epi(site: Site, fault: Fault) -> float:
    """Calculate 2d distance between site and hypocenter."""
    return jnp.linalg.norm(site.calc_xy() - fault.calc_xy_hyp(), ord=2)

# xyz Distance from hypocenter
def calc_R_hyp(site: Site, fault: Fault) -> float:
    """Calculate 3d distance between site and hypocenter."""
    return jnp.linalg.norm(site.calc_xyz() - fault.calc_xyz_hyp(), ord=2)

# xy Distance from top of rupture
def calc_R_x(site: Site, fault: Fault) -> float:
    """Calculate 2d distance between site and top of rupture."""
    fault_xyz_tor = fault.calc_xyz_hyp() + fault.calc_dxyz_tor()
    return jnp.linalg.norm(site.calc_xyz() - fault_xyz_tor, ord=2)

def calc_R(site:Site, fault:Fault):
    """Calculate distances of interest."""
    return [calc_R_jb(site, fault), calc_R_rup(site, fault), calc_R_epi(site, fault), calc_R_hyp(site, fault), calc_R_x(site, fault)]
    
@jtu.register_pytree_node_class
class Scenario:
    def __init__(self, site:Site, fault_tree:Fault):
        """
        Initialize a Scenario object.

        Args
        ----------
        site : Site
            A single site.
        self.fault_tree:
            A vectorized fault. 
        """
        self.site = site
        self.fault_tree = fault_tree

    def calc_fault_num(self):
        return self.fault_tree()
    
    def calc_R_jb(self):
        R_jbs = jax.vmap(jtu.Partial(calc_R_jb, self.site))(self.fault_tree)
        return R_jbs
    
    def calc_R_rup(self):
        R_rups = jax.vmap(jtu.Partial(calc_R_rup, self.site))(self.fault_tree)
        return R_rups
    
    def calc_R_epi(self):
        R_epis = jax.vmap(jtu.Partial(calc_R_epi, self.site))(self.fault_tree)
        return R_epis
    
    def calc_R_hyp(self):
        R_hyps = jax.vmap(jtu.Partial(calc_R_hyp, self.site))(self.fault_tree)
        return R_hyps

    def calc_R_x(self):
        R_xs = jax.vmap(jtu.Partial(calc_R_x, self.site))(self.fault_tree)
        return R_xs
    
    def tree_flatten(self):
        return (self.site, self.fault_tree), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
    
@jtu.register_pytree_node_class
class GMMLT:
    def __init__(self, gmms:list, T:float, weights:jax.typing.ArrayLike):
        self.T = T
        self.gmms = gmms
        self.weights = weights

    # Calculate for a single GMM. Takes R so we don't repeat the calculation every time.
    def calc_single(self, i:int, Mw:float, site:Site, fault:Fault, R:jax.Array):
        return lax.switch(i, self.gmms, Mw, self.T, site, fault, R)

    def tree_flatten(self):
        return (self.T, self.weights), self.gmms
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux, *children)

def calc_haz(x:float, M_min:float, gmms:GMMLT, scn:Scenario, dM:float = 0.1):
    fault_num = scn.fault_tree.x.shape[0]
    M_min, M_max = scn.fault_tree.mfd.M_min, scn.fault_tree.mfd.M_max

    # Magnitude bins.
    bins_M = jnp.arange(M_min.min(), M_max.max() + dM, dM)
    # MFD rates
    n_M = jax.vmap(scn.fault_tree.mfd.calc_lmdaM)(bins_M)
    # Take difference to make incremental
    n_M_inc = n_M[:-1] - n_M[1:]

    # Roots for GMM evaluation.
    roots_M = (bins_M[:-1] + bins_M[1:]) / 2
    # Array of ones/zeros for each fault signifying array inside/outside range (shape (roots_M.shape, fault_num))
    weights_mask = (roots_M[:, None] > M_min[None, :]) & (roots_M[:, None] < M_max[None, :])
    # We can think of our incremental rates as already including quadrature weights, so 
    #   we'll just multiply them by the mask to be safe. 
    n_M_inc = n_M_inc * weights_mask

    # Ground motion means + stds for each fault at roots
    # Calculate R for full fault tree...
    R_tree = jax.vmap(calc_R, in_axes = (None, 0))(scn.site, scn.fault_tree)
    # Triple vmap. First, across faults (and corresponding distances),
    calc_faults = jax.vmap(gmms.calc_single, in_axes=(None, None, None, 0, 0))
    # Then across magnitudes,
    calc_M = jax.vmap(calc_faults, in_axes=(None, 0, None, None, None))
    # Then across GMMs. This order minimizes recompilation.
    calc_gmms = jax.vmap(calc_M, in_axes=(0, None, None, None, None))
    # Grab indices and vmap across
    gmm_idcs = jnp.arange(len(gmms.gmms))
    all_mu_lnSA, all_std_lnSA = calc_gmms(gmm_idcs, roots_M, scn.site, scn.fault_tree, R_tree)
    # Get PoE at all points
    all_prob_x = 1 - trunc_norm_cdf(jnp.log(x), 
                                    a = -jnp.inf, b = 3 * all_std_lnSA, 
                                    loc = all_mu_lnSA, scale = all_std_lnSA)

    # Take mean
    mu_prob_x = jnp.einsum('i,ijk->jk', gmms.weights, all_prob_x)

    # Hazard integrand (magnitude probabilities * exceedance probabilities)
    haz_intgrnd = mu_prob_x * n_M_inc

    # Return the sum since our "quadrature weights" are basically
    #   included in our frequency bins
    return haz_intgrnd.sum()