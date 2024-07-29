import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'


import jax
import jax.numpy as jnp
import flax
import optax
from flax.training.train_state import TrainState
from functools import partial
from tqdm import tqdm

from datasets.mnist import *
from protocol_train import *
from model.resnet_v4 import *

num_epochs = 3
batch_size = 32
nonlinearity = nn.relu
mnmx = [-4, 0, -4, 0]
resolution = 8

# Prepare dataset
train_ds, test_ds = prepare_dataset(batch_size)
total_batch = train_ds.cardinality().numpy()

@partial(jax.pmap, in_axes=(None, 0, None), static_broadcasted_argnums=(0, 2))
@partial(jax.vmap, in_axes=(None, 0, None))
def train_step_v(variables, lr, model):
    state = TrainState.create(
        apply_fn=model.apply,
        params=jax.tree_map(lambda param: param + lr[0], variables['params']),
        batch_stats=variables['batch_stats'],
        tx=optax.sgd()
    )

    M_train = []
    for _ in tqdm(range(num_epochs), total=num_epochs, leave=False, desc='Epochs'):
        for batch in tqdm(train_ds.as_numpy_iterator(), total=total_batch, leave=False, desc='Iter'):
            state, metrics = train_step(state, batch)
        M_train.append(metrics)

    return M_train


# Model loading
for batch in train_ds.as_numpy_iterator():
    x = batch['image']
    y = batch['label']
    break
resnet20 = ResNet(10, nonlinearity, ResNetBlock)
variables = resnet20.init(jax.random.PRNGKey(1), x)

lrs = scaling_sketch(mnmx, resolution).reshape((8, 8, 2))
outputs = train_step_v(variables, lrs, resnet20)



@partial(jax.pmap, in_axes=0)
@partial(jax.vmap, in_axes=0)
def test(x):
    print(x.shape)
    return x

test(jnp.ones((8, 12, 28, 28, 1))).shape