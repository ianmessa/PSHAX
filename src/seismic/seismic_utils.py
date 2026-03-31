import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import tree as jt
from jax.scipy.stats.truncnorm import cdf as trunc_norm_cdf

from numerics import *

# Calculate SOF from rake
def _rake_SOF_flag(rake:float):
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

# Magnitude-frequency distribution
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
        return _rake_SOF_flag(self.rake)

    def tree_flatten(self):
        return (self.x, self.y, self.z_hyp, self.z_tor, self.theta, self.dip, self.rake, self.width, self.HW_flag, self.mfd), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# Logic tree
@jtu.register_pytree_node_class
class GMMLT:
    """
    GMM Logic Tree (GMMLT)

    This class represents a logic tree for combining multiple Ground Motion Models (GMMs) 
    with associated weights for probabilistic seismic hazard analysis.

    Attributes
    ----------
    T : float
        The spectral period for which the ground motion is evaluated.
    gmms : list
        A list of GMM callable objects. Each GMM should accept the arguments 
        (Mw, T, site, fault, R) and return (mu_lnSA, sigma_lnSA).
    w : jax.typing.ArrayLike
        An array of weights corresponding to each GMM in the logic tree.

    Methods
    -------
    calc_single(i: int, Mw: float, site: Site, fault: Fault, R: jax.Array)
        Computes the mean and standard deviation of lnSA for a single GMM specified by index i.

    tree_flatten()
        Flattens the GMMLT object for JAX pytree compatibility.

    tree_unflatten(aux, children)
        Reconstructs a GMMLT object from flattened components for JAX pytree compatibility.
    """
    def __init__(self, gmms:list, w:jax.typing.ArrayLike):
        self.gmms = gmms
        self.w = w

    def calc_single(self, i:int, Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
        """
        Computes the mean and standard deviation of the natural logarithm of spectral acceleration (lnSA)
        for a single Ground Motion Model (GMM) specified by its index.

        Parameters
        ----------
        i : int
            Index of the GMM in the logic tree to use for the calculation.
        Mw : float
            Moment magnitude of the earthquake.
        site : Site
            Site object containing site-specific parameters.
        fault : Fault
            Fault object containing fault-specific parameters.
        R : jax.Array
            Array of distances from the site to the fault.

        Returns
        -------
        mu_lnSA : float
            Mean of the natural logarithm of spectral acceleration.
        sigma_lnSA : float
            Standard deviation of the natural logarithm of spectral acceleration.
        """
        return lax.switch(i, self.gmms, Mw, T, site, fault, R)
    
    def calc_all(self, Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
        gmm_idcs = jnp.arange(len(self.gmms))
        return jax.vmap(self.calc_single, in_axes = (0, None, None, None, None, None))(gmm_idcs, Mw, T, site, fault, R)
    
    def calc_median(self, Mw:float, T:float, site:Site, fault:Fault, R:jax.Array):
        all_mu_lnSA, all_std_lnSA = self.calc_all(Mw, T, site, fault, R)
        median_mu_lnSA = all_mu_lnSA @ self.w
        # Mixture model...
        median_std_lnSA = ((all_mu_lnSA**2 + all_std_lnSA**2) @ self.w - median_mu_lnSA ** 2) ** (1 / 2)
        return median_mu_lnSA, median_std_lnSA

    def tree_flatten(self):
        return (self.w,), self.gmms
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux, *children)

# Distances...
# xy Distance from nearest edge of rupture
def _calc_R_jb(site: Site, fault: Fault) -> float:
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
def _calc_R_rup(site: Site, fault: Fault) -> float:
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
def _calc_R_epi(site: Site, fault: Fault) -> float:
    """Calculate 2d distance between site and hypocenter."""
    return jnp.linalg.norm(site.calc_xy() - fault.calc_xy_hyp(), ord=2)

# xyz Distance from hypocenter
def _calc_R_hyp(site: Site, fault: Fault) -> float:
    """Calculate 3d distance between site and hypocenter."""
    return jnp.linalg.norm(site.calc_xyz() - fault.calc_xyz_hyp(), ord=2)

# xy Distance from top of rupture
def _calc_R_x(site: Site, fault: Fault) -> float:
    """Calculate 2d distance between site and top of rupture."""
    fault_xyz_tor = fault.calc_xyz_hyp() + fault.calc_dxyz_tor()
    return jnp.linalg.norm(site.calc_xyz() - fault_xyz_tor, ord=2)

def calc_R(site:Site, fault:Fault):
    """Calculate distances of interest."""
    return [_calc_R_jb(site, fault), _calc_R_rup(site, fault), _calc_R_epi(site, fault), _calc_R_hyp(site, fault), _calc_R_x(site, fault)]

# Wrapper for site-fault pairs    
@jtu.register_pytree_node_class
class Scenario:
    def __init__(self, M:float, T:float, site:Site, fault:Fault):
        self.M = M
        self.T = T
        self.site = site
        self.fault = fault
    
    def calc_R(self):
        return calc_R(self.site, self.fault)
    
    def tree_flatten(self):
        return ((self.M, self.T, self.site, self.fault), None)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
    
    def tree_tovec(self):
        # Flatten all objects
        site_vec = self.site.tree_flatten()[0]
        # (Omitting x, y, and MFD from fault)
        fault_vec = self.fault.tree_flatten()[0][2:-1]
        mfd_vec = self.fault.mfd.tree_flatten()[0]
        # Convert fault to polar relative to site
        site_fault_dx = self.fault.x - self.site.x
        site_fault_dy = self.fault.y - self.site.y
        fault_R = (site_fault_dx**2 + site_fault_dy**2) ** (1 / 2)
        fault_theta = jnp.atan2(site_fault_dy, site_fault_dx)
        # Wrap
        scn_vec = jnp.array((self.M, self.T,) + site_vec + (fault_R, fault_theta) + fault_vec + mfd_vec)
        return scn_vec

    @classmethod
    def objs_fromvec(cls, scn_vec):
        M, T = scn_vec[0:2]
        site_xy = scn_vec[2:4]
        site_vec = scn_vec[4:8]
        fault_R, fault_theta = scn_vec[8:10]
        fault_vec = scn_vec[10:17]
        mfd_vec = scn_vec[17:]
        # Bounce fault polar coordinates back to Cartesian
        fault_xy = site_xy + fault_R * jnp.array([jnp.cos(fault_theta), jnp.sin(fault_theta)])
        return M, T, Site(*site_xy, *site_vec), Fault(*fault_xy, *fault_vec, MFD(*mfd_vec))
    
    @classmethod
    def tree_fromvec(cls, scn_vec):
        return cls(*cls.objs_fromvec(scn_vec))
