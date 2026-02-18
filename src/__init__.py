import os
import jax
if os.environ.get("JAX_ENABLE_X64") is None:
    jax.config.update("jax_enable_x64", True)
from . import numerics, seismic