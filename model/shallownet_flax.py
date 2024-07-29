import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4'

# Computation settings
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax

from typing import Any, List

class ShallowNet(nn.Module):
    width: int = 300
    hidden_depth: int = 1
    actfn: Any = nn.relu
    rng = jax.random.key(42)
    
    @nn.compact
    def __call__(self, x):
        layer_in2hid = nn.Dense(self.width, kernel_init=nn.initializers.xavier_normal(), use_bias=False)
        layer_hid2hid = nn.Dense(self.width, kernel_init=nn.initializers.xavier_normal(), use_bias=False)
        layer_hid2out = nn.Dense(10, kernel_init=nn.initializers.xavier_normal(), use_bias=False)

        # flatten
        x = x.reshape((x.shape[0], -1))

        # in-hidden
        x = layer_in2hid(x)
        x = self.actfn(x)
        # hidden-hidden
        for _ in range(self.hidden_depth):
            x = layer_hid2hid(x)
            x = self.actfn(x)
        # hidden-out
        x = layer_hid2out(x)
        return nn.softmax(x)



if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')

    def data_normalize(ds):
        return ds.map(lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255.,
            'label': sample['label']
        })

    train_ds = data_normalize(train_ds).shuffle(buffer_size=10, seed=42).batch(100).prefetch(1).take(1000)
    test_ds = data_normalize(test_ds).shuffle(buffer_size=10, seed=42).batch(100).prefetch(1).take(1000)

    total_batch = train_ds.cardinality().numpy()
    total_tbatch = test_ds.cardinality().numpy()

    x = jnp.ones((100, 28, 28, 1))
    rng = jax.random.key(42)
    mnmx = [-6, 6, -6, 6]
    
    # dense = nn.Dense(300)
    # dense.kernel_init = nn.initializers.xavier_normal()(rng, (100, 300)) * 1e-6
    # print(dense.kernel_init)

    snet = ShallowNet()
    variables = snet.init(rng, x)
    
    # lrband
    res = 32
    lr0 = jnp.logspace(-3, 6, res)
    lr1 = jnp.logspace(-3, 6, res)
    xx, yy = jnp.meshgrid(lr0, lr1)
    xx = xx.reshape((-1))
    yy = yy.reshape((-1))
    lrs = jnp.stack((xx, yy), axis=1)

    def crate_state(lrs):
        state = train_state.TrainState.create(
            apply_fn=snet.apply,
            params=variables['params'],
            tx=optax.sgd(lrs[0])
        )
        state = state.replace(
            params={
                'Dense_0': {'kernel': state.params['Dense_0']['kernel'] * 10**mnmx[0] + lrs[1]},
                'Dense_1': {'kernel': state.params['Dense_1']['kernel'] * 10**mnmx[0] + lrs[1]},
                'Dense_2': {'kernel': state.params['Dense_2']['kernel'] * 10**mnmx[0]}
            }
        )
        return state
    
    states = [crate_state(lr) for lr in lrs]
    
    from functools import partial
    @jax.jit
    # @partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
    def train_step(states, batch):
        def loss_fn(params):
            logits = states.apply_fn({'params': params}, x=batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
            return loss, logits
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(states.params)
        states = states.apply_gradients(grads=grads)
        return states, loss
    
    from tqdm import tqdm
    loss_archive = []
    for i in tqdm(range(len(states)), total=len(states), leave=False):
        for i in range(100):
            losses = []
            for batch in train_ds.as_numpy_iterator():
                states[i], loss = train_step(states[i], batch)
                losses.append(loss)
        loss_archive.append(np.array(losses))

    print(loss_archive)

