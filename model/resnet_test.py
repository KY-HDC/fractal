import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import optax
import numpy as np
from functools import partial


# GPSMD
# mesh = Mesh(np.array(jax.devices()).reshape((2, 2)), ('x', 'y'))

# Model
def net(theta, X):
    '''Need to jit! Beware of using if-condition.'''
    Z = X.reshape((X.shape[0], -1))
    for W in theta[:-1]:
        Z = jnp.dot(Z, W)
        Z = jax.nn.relu(Z)
    Z = jnp.dot(Z, theta[-1])
    return jax.nn.softmax(Z)

# Weight initialization
def init(rng, width, layers=34, initializer=jax.nn.initializers.he_normal(), init_amp=1e-6):
    
    key = jax.random.key(rng)
    rngs = jax.random.split(key, layers)
    
    # theta
    theta = [
        initializer(rng, (784, width)) * init_amp if i == 0
        else initializer(rng, (width, width)) * init_amp
        for i, rng in enumerate(rngs[:-1])
    ] + [initializer(rngs[-1], (width, 10)) * init_amp]
    
    # # batch-normalization (COMING SOON)
    # key = jax.random.key(rng+1)
    # rngs = jax.random.split(key, hidden+1)
    # bn = [
    #     (initializer(rng, (1, 1)) * init_amp) * 2
    #     for rng in rngs
    # ]
    return theta

# Evaluate loss
def loss(theta, X, Y):
    return optax.softmax_cross_entropy_with_integer_labels(
        jnp.clip(net(theta, X), 1e-10, 1.), Y
        ).mean()

# One-step training
@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None), out_axes=(0, 0, 0))
def train_step(theta, hparams, batch):
    X = batch['image']
    Y = batch['label']

    theta[0] += hparams[0]
    learning_rates = [hparams[1]] * 2
    
    @partial(jax.jit,
             in_shardings=(
                 NamedSharding(mesh, PartitionSpec(None, 'y')),
                 NamedSharding(mesh, PartitionSpec('x', None)),
                 NamedSharding(mesh, PartitionSpec()),
                 NamedSharding(mesh, PartitionSpec())
             ))     # if using BN, BN can't be jitting.
    def updates(theta, X, Y, learning_rates):
        Z = net(theta, X)
        _acc = Z.argmax(axis=-1) == Y
        _acc = _acc.sum() / _acc.shape[0]
        
        _loss, _grad = jax.value_and_grad(loss)(theta, X, Y)
        return jax.tree_map(lambda t, g, lr: t - lr * g, theta, _grad, learning_rates), _loss, _acc
    return updates(theta, X, Y, learning_rates)

# One-step validation
def eval_step_px(theta, tbatch):
    X = tbatch['image']
    Y = tbatch['label']
    _logits = net(theta, X)
    _acc = _logits.argmax(axis=-1) == Y
    _acc = _acc.sum() / _acc.shape[0]
    _loss = loss(theta, X, Y)
    return _logits, _loss, _acc

eval_step = jax.jit(jax.vmap(eval_step_px, in_axes=(0, None), out_axes=(0, 0, 0)))
jitted_eval_step_px = jax.jit(eval_step_px)

# @jax.jit
# @partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0, 0))
# def eval_step(theta, tbatch):
#     X = tbatch['image']
#     Y = tbatch['label']
#     _logits = net(theta, X)
#     _acc = _logits.argmax(axis=-1) == Y
#     _acc = _acc.sum() / _acc.shape[0]
#     _loss = loss(theta, X, Y)
#     return _logits, _loss, _acc


if __name__ == "__main__":
    
    # initialization
    x = jax.nn.initializers.xavier_normal()(jax.random.key(42), (15, 28, 28, 1))     # NHWC
    W1 = jax.nn.initializers.xavier_normal()(jax.random.key(421), (3, 3, 1, 16))      # HWIO (C=I -> O)
    W2 = jax.nn.initializers.xavier_normal()(jax.random.key(421), (3, 3, 16, 32))      # HWIO (C=I -> O)
    W3 = jax.nn.initializers.xavier_normal()(jax.random.key(421), (3, 3, 32, 64))      # HWIO (C=I -> O)

    # net(stage1)
    output = jax.lax.conv(jnp.transpose(x, [0, 3, 1, 2]), jnp.transpose(W1, [3, 2, 0, 1]), window_strides=(1, 1), padding='SAME')
    output = jax.lax.conv(output, jnp.transpose(W2, [3, 2, 0, 1]), window_strides=(1, 1), padding='SAME')
    output = jax.lax.conv(output, jnp.transpose(W3, [3, 2, 0, 1]), window_strides=(1, 1), padding='SAME')
    output = output + np.transpose(x, [0, 3, 1, 2])     # skip-connection

    print(output.shape)

    ### arch. structure
    # init: make all weights about ResNet-34