import jax
import jax.numpy as jnp
import numpy as np
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4'

x = jnp.ones((32, 5, 2))
y = jnp.ones((32, 2, 3))

def shard_data(data, n_devices):
    data = data.reshape(n_devices, data.shape[0] // n_devices, *data.shape[1:])
    return data

def f(x, y):
    return x.dot(y)

# f = jax.vmap(f, in_axes=(0,0), out_axes=0)
f = jax.pmap(f, in_axes=(0, 0), out_axes=0)
f = jax.pmap(jax.vmap(f, in_axes=(0, 0)))
x = shard_data(x, 4)
y = shard_data(y, 4)

output = f(x, y)
print(output.shape)
