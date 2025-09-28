import jax
from jax import numpy as jnp
from jax import random as jrnd
from matplotlib import pyplot as plt
from scipy.interpolate import RBFInterpolator
# title: 1, exp(xy[:, 1] ** 2)
# sub 1: 1, exp(xy[:, 1] ** (1 / 2))
    # body 1: 1, exp(xy[:, 1])
# sub 2: 3, same as sub 1
    # body 2: 3, same as body 1
# sub 3: 2
    # body 3: 2
# sub 4: 4

# Crop 2 units on all sides UPON IMPORT
def generate_background(key_num, exp_c:int = 1):
    m = 800 * 1.5
    n = 450 * 1.5
    m, n = int(m), int(n)

    xyz = jrnd.uniform(jrnd.key(key_num), (150, 3))
    xyz = xyz.at[:, 1].set(xyz[:, 1] * (m / n))
    xy, z = xyz[:, :2], xyz[:, 3]
    # CHANGES WITH BACKGROUND #
    z = z * (jnp.exp(xy[:, 1] ** exp_c) - 1)
    x_full = jnp.linspace(0, m / n, m)
    y_full = jnp.linspace(0, 1, n)
    xy_full = jnp.stack(jnp.meshgrid(y_full, x_full), axis = -1).reshape(n * m, 2)
    z2 = RBFInterpolator(xy, z)(xy_full).reshape(m, n)

    fig = plt.figure(frameon=False, dpi = 300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(z2.T, cmap = 'jet')
    plt.savefig('slide_background_seed=%s_c=%s.png'%(key_num, exp_c))
    plt.close()