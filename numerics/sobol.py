import jax
from jax import lax
from jax import tree_util as jtu
from jax import numpy as jnp
from jax import random as jrnd

import polars as pl
from pathlib import Path

# Setup
sobol_1111_path = Path(__file__).parent / "sobol_1111.tsv"
dir_nums = pl.read_csv(sobol_1111_path, separator = '\t', has_header = False).to_jax()
dim_max = 1111
bits_max = 30

# Output array
V = jnp.zeros((dim_max, bits_max), dtype = int)
poly = dir_nums[:, 0].astype('uint32')
V = V.at[:, :13].set(dir_nums[:, 1:14])
V = V.at[0, :].set(1)

# Binary representations
binary_repr = jnp.unpackbits(poly.view('uint8'), bitorder = 'little').reshape(dim_max, 32)[:, ::-1]
# Representation lengths
m = jax.vmap(jtu.Partial(jnp.flatnonzero, size = 32))(a = binary_repr[:, ::-1]).max(axis = 1)

# Binary inclusion indices (just set first ones to zeros)
binary_incl = binary_repr.at[jnp.arange(dim_max), 31 - m].set(0)

# Single bitwise update
def _update_V_ijk(k, state):
    # Unpack
    j, V_i, m_i, binary_incl_i = state
    # Get inclusion condition
    incl_ik = lax.dynamic_index_in_dim(binary_incl_i, k, keepdims = False)

    # k = 31 -> kk = m_i - 1
    kk = k - 32 + m_i
    # Get entries from V
    V_ij = lax.dynamic_index_in_dim(V_i, j, keepdims = False)
    V_ijmkm1 = lax.dynamic_index_in_dim(V_i, j - kk - 1, keepdims = False)

    # Bitwise xor & selection
    V_ij_xor = jnp.bitwise_xor(V_ij, 2 ** (kk + 1) * V_ijmkm1)
    V_i = lax.select(incl_ik, V_i.at[j].set(V_ij_xor), V_i)

    return j, V_i, m_i, binary_incl_i

# Single-element update
def _update_V_ij(j, state):
    # Unpack
    V_i, m_i, binary_incl_i = state

    # m-periodic entry from V
    V_ijmm = lax.dynamic_index_in_dim(V_i, j - m_i, keepdims = False)

    # Set initial state for bitwise update
    init_xor_state = (j, V_i.at[j].set(V_ijmm), m_i, binary_incl_i)
    # Repeat bitwise update for all bits in inclusion
    V_i_periodic_bitwise = lax.fori_loop(0, 32, _update_V_ijk, init_xor_state)[1]

    # Bool selecion for periodicity after we iterate
    V_i = lax.select(j >= m_i, V_i_periodic_bitwise, V_i)

    return V_i, m_i, binary_incl_i

# Full row update
def _update_V_i(V_i, m_i, binary_incl_i):
    # Easy. Element update for each row.
    init_row_state = V_i, m_i, binary_incl_i
    return lax.fori_loop(0, bits_max, _update_V_ij, init_row_state)[0]

# Update our whole array using fns above
V = jax.vmap(_update_V_i)(V, m, binary_incl)

def _owen_scramble(x_uint32, seed):
    # Ensure starting point and seed are uint32
    x_uint32 = jnp.uint32(x_uint32)
    seed = jnp.uint32(seed)

    def body_fn(i, val):
        # 1. Create a mask for the i-th bit (MSB is 0)
        # Using uint32 for shift prevents sign-bit overflow
        mask = jnp.uint32(0x80000000) >> i
        
        # 2. Extract higher bits to seed the permutation of the current bit
        # This is the "Nested" part of Owen's scrambling
        shift = jnp.uint32(31) - i + jnp.uint32(1)
        hash_input = lax.select(i > 0, jnp.right_shift(val, shift), jnp.uint32(0))
        hash_input = hash_input ^ seed
        
        # 3. Simple, JAX-friendly mixing hash (Low-bias XOR-Shift)
        h = hash_input
        h ^= h >> 16
        h *= jnp.uint32(0x85ebca6b)
        h ^= h >> 13
        h *= jnp.uint32(0xc2b2ae35)
        h ^= h >> 16
        
        # 4. Flip the bit at mask position if the hash LSB is 1
        flip = jnp.where(h & jnp.uint32(1), mask, jnp.uint32(0))
        return val ^ flip

    return lax.fori_loop(0, 32, body_fn, x_uint32)

def sobol(n:int, d:int): 
    """
    Draw samples from Sobol sequence.

    Parameters
    ----------
    dim : int
        The number of dimensions for the Sobol sequence.
    n : int
        The number of samples to generate.
    Returns
    -------
    points : jax.Array
        An array of shape (n, dim) containing samples from Sobol sequence.
    Notes
    -----
    - We use 32-bit representations for sample indices. You'll be okay as long as
        n < 4e9.
    """
    # Scale
    V_dim = V[:d] * 2 ** jnp.arange(bits_max)[::-1]

    # Calculate rightmost zeros for sampling
    n_range = jnp.arange(n).astype('uint32')
    n_rmz = jnp.log2(jnp.bitwise_and(n_range, -n_range)).astype(int).at[0].set(0)

    # Sample by recursively updating array of zeros
    def sample(point_i, rmz_i):
        V_rmz = lax.dynamic_index_in_dim(V_dim, index = rmz_i, axis = 1, keepdims = False)
        point_ip1 = jnp.bitwise_xor(point_i, V_rmz)
        return point_ip1, point_ip1 / 2 ** bits_max
    
    init_sample_state = jnp.zeros(d, dtype = int)
    _, x = lax.scan(sample, init_sample_state, n_rmz)

    return x

def sobol_scrambled(n:int, d:int, key:jax.Array = jrnd.key(0)): 
    """
    Draw samples from Sobol sequence.

    Parameters
    ----------
    dim : int
        The number of dimensions for the Sobol sequence.
    n : int
        The number of samples to generate.
    key: jax.Array
        Random number generator
    Returns
    -------
    points : jax.Array
        An array of shape (n, dim) containing samples from a scrambled Sobol sequence.
    Notes
    -----
    - We use 32-bit representations for sample indices. You'll be okay as long as
        n < 4e9.
    """
    # Scale
    V_dim = V[:d] * 2 ** jnp.arange(bits_max)[::-1]

    # Calculate rightmost zeros for sampling
    n_range = jnp.arange(n).astype('uint32')
    n_rmz = jnp.log2(jnp.bitwise_and(n_range, -n_range)).astype(int).at[0].set(0)

    # Sample by recursively updating array of zeros
    def sample(point_i, rmz_i):
        V_rmz = lax.dynamic_index_in_dim(V_dim, index = rmz_i, axis = 1, keepdims = False)
        point_ip1 = jnp.bitwise_xor(point_i, V_rmz)
        return point_ip1, point_ip1
    
    init_sample_state = jnp.zeros(d, dtype = int)
    _, x_uint32 = lax.scan(sample, init_sample_state, n_rmz)

    keys = jrnd.split(key, d)
    keys_uint32 = jax.vmap(jrnd.bits)(keys)
    
    scramb = jax.vmap(jax.vmap(_owen_scramble, in_axes = (0, None)), in_axes = (1, 0))
    x = scramb(x_uint32, keys_uint32).astype(jnp.float32) / (2.**32)
    x = x.astype(jnp.float_)
    # Scale back to 0, 1
    x = (x - x.min(axis = 1, keepdims = True)) * 4

    return x.T