import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

x = {
    'a': jnp.array([1, 2, 3]*4).reshape((4, 3)),
    'b': jnp.array([4, 5, 6]*4).reshape((4, 3)),
    'c': jnp.array([7, 8, 9]*4).reshape((4, 3))
}

y = jnp.array([0.01, 0.02, 0.03, 0.04]).reshape((4,))

@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def f(x, y):
    return jax.tree_map(lambda x: x+y, x)

print(f(x, y))

