import jax
import jax.numpy as jnp
from timeit import timeit

N = 10
I, O = 100, 150

key1, key2 = jax.random.split(jax.random.PRNGKey(42))
mat = jax.random.normal(key1, (O, I))
batched_x = jax.random.normal(key2, (N, I))

def apply_matrix(x):
    return jnp.dot(mat, x)

# Naive
def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
t_naive = timeit("naively_batched_apply_matrix(batched_x).block_until_ready()", globals=globals(), number=1000)

# jitting
import numpy as np

@jax.jit
def batched_apply_matrix(batched_x):
  return jnp.dot(batched_x, mat.T)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Manually batched')
t_jit = timeit("batched_apply_matrix(batched_x).block_until_ready()", globals=globals(), number=1000)

# jitting with vmap
@jax.jit
def vmap_batched_apply_matrix(batched_x):
  return jax.vmap(apply_matrix)(batched_x)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           vmap_batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Auto-vectorized with vmap')
t_vmap = timeit("vmap_batched_apply_matrix(batched_x).block_until_ready()", globals=globals(), number=1000)

print(t_naive, t_jit, t_vmap)


jnp.repeat()