import jax 
from jax import random as jrnd
from jax import numpy as jnp

from matplotlib import pyplot as plt

from numerics.sobol import *

jax.config.update('jax_enable_x64', True)

def pts_mc(a:float|jax.Array, b:float|jax.Array, n:int, d:int, key:jax.Array = jrnd.key(0)):
    a = jnp.broadcast_to(a, (d,))
    b = jnp.broadcast_to(b, (d,))
    x = jrnd.uniform(key, (n, d), minval = a, maxval = b)
    return x

def pts_rqmc(a:float|jax.Array, b:float|jax.Array, n:int, d:int, key:jax.Array = jrnd.key(0)):
    a = jnp.broadcast_to(a, (d,))
    b = jnp.broadcast_to(b, (d,))
    x = sobol_scrambled(n, d, key)
    x = x * (b - a) + a
    return x