import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.pmap, axis_name='rows')
@partial(jax.pmap, axis_name='cols')
def normalize(x):
    row_normed = x / jax.lax.psum(x, 'rows')
    col_normed = x / jax.lax.psum(x, 'cols')
    doubly_normed = x / jax.lax.psum(x, ('rows', 'cols'))
    return row_normed, col_normed, doubly_normed

f = lambda x: x + jax.lax.psum(x, axis_name='i')
data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
out = jax.pmap(f, axis_name='i')(data)  # doctest: +SKIP
print(out)  # doctest: +SKIP