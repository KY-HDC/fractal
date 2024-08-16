# For running test
import os; os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from tqdm import tqdm

if __name__ == "__main__":
    from flax_resnet import *
else:
    from model.flax_resnet import *

from protocol_args import nonlinearity
leaky = False if nonlinearity=="relu" else True

'''
This code will run resnet from the weights built by FLAX.
'''

# Initialization
def initialize(module, rng, x, amp=1.):
    variables = module.init(jax.random.PRNGKey(rng), x)
    variables['params']['Dense_0']['kernel'] = \
        jax.nn.initializers.xavier_normal()(jax.random.PRNGKey(1), (50176, 10))  # 64 * 28**2
    
    def conv_dog(kp, x):
        kp = jax.tree_util.keystr(kp)
        if 'Conv' in kp:
            x = x * amp
            x = jnp.transpose(x, (3, 2, 0, 1))
        return x
    
    variables['params'] = jax.tree_util.tree_map_with_path(conv_dog, variables['params'])
    variables['batch_stats'] = jax.tree_map(lambda stats: stats.reshape((1, stats.shape[0], 1, 1)), variables['batch_stats'])

    return variables


# Resnet's architecture
def net(variables, x: jnp.array, on_train=True):
    '''This function returns logit and variable(weight).'''

    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # input.T: NHWC -> NCHW
    x = jnp.transpose(x, [0, 3, 1, 2])
    print(x.shape)
    # 1st conv
    x = jax.lax.conv(x, params['Conv_0']['kernel'], window_strides=(1, 1), padding='SAME')
    x, batch_stats['BatchNorm_0'] = batchnorm(x, params['BatchNorm_0'], batch_stats['BatchNorm_0'], on_train=on_train)
    x = jax.lax.cond(leaky, nn.leaky_relu, nn.relu, x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')

    # ResNetBlocks; conv0-conv1-skip
    for k, v in params.items():
        if 'ResNetBlock' in k:

            residual = x

            k_conv0 = v['Conv_0']['kernel']
            x = jax.lax.conv(x, k_conv0, window_strides=(1, 1), padding='SAME')
            x, batch_stats[k]['BatchNorm_0'] = batchnorm(x, v['BatchNorm_0'], batch_stats[k]['BatchNorm_0'], on_train=on_train)
            x = jax.lax.cond(leaky, nn.leaky_relu, nn.relu, x)
            
            k_conv1 = v['Conv_1']['kernel']
            x = jax.lax.conv(x, k_conv1, window_strides=(1, 1), padding='SAME')
            x, batch_stats[k]['BatchNorm_1'] = batchnorm(x, v['BatchNorm_1'], batch_stats[k]['BatchNorm_1'], on_train=on_train)
            
            if 'Conv_2' in v.keys():
                k_conv2 = v['Conv_2']['kernel']
                residual = jax.lax.conv(residual, k_conv2, window_strides=(1, 1), padding='SAME')
            x += residual
            x = jax.lax.cond(leaky, nn.leaky_relu, nn.relu, x)

    # FC
    x = nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    x = jnp.transpose(x, [0, 2, 3, 1])
    x = x.reshape((x.shape[0], -1))
    x = jnp.dot(x, params['Dense_0']['kernel'])

    # Memorizing batch_stats
    variables = {'params': params, 'batch_stats': batch_stats}

    return nn.softmax(x), variables


# Resnet's subfunctions 1: Loss function
# @partial(jax.jit, static_argnums=3)
def loss_fn(variables, x, y, on_train=True):
    logits, variables = net(variables, x, on_train=on_train)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), (logits, variables)


# Resnet's subfunctions 2: Update function
# @jax.jit
# TODO: How about optax?
# TODO: Batches <- All devices must use same batches.
# TODO: Weights <- Each device uses 1/4 of models.
# @partial(jax.pmap, in_axes=(0, None, None, 0))
@partial(jax.vmap, in_axes=(0, None, None, 0))
@jax.jit
def update_fn(variables, x, y, lr):
    (loss, (logits, variables)), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(variables, x, y)    
    variables['params'] = jax.tree_map(
        lambda param, l, g: \
            param - l * g, variables['params'], lr, grads['params']
            )
    
    return variables, (loss, logits)


# Resnet's subfunctions 3: Batch normalization function
def batchnorm(x, params_bn, batch_stats_bn, momentum=0.9, eps=1e-6, on_train=True):
    '''Batch normalizing
        *Args
            params: variables['params']['BatchNorm_X']
            batch_stats: variables['batch_stats']['BatchNorm_X']
    '''
    gamma = params_bn['scale']
    beta = params_bn['bias']
    gamma = gamma.reshape((1, gamma.shape[0], 1, 1))
    beta = beta.reshape((1, beta.shape[0], 1, 1))

    running_mu = batch_stats_bn['mean']
    running_var = batch_stats_bn['var']
    
    def mode_train():
        mu = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
        r_mu = momentum * running_mu + (1 - momentum) * mu
        r_var = momentum * running_var + (1 - momentum) * var
        return (x - mu) / jnp.sqrt(var + eps), r_mu, r_var
    
    def mode_inference():
        r_mu = running_mu
        r_var = running_var
        return (x - r_mu) / jnp.sqrt(r_var + eps), r_mu, r_var
        
    x, running_mu, running_var = jax.lax.cond(on_train, mode_train, mode_inference)
    
    x = gamma * x + beta

    batch_stats_bn['mean'] = running_mu
    batch_stats_bn['var'] = running_var

    return x, batch_stats_bn


# Combinate learning rate and offset, and synchronize their tree structure.
@partial(jax.vmap, in_axes=(0, 0), out_axes=(0, 0))     # induced trace; makes error; tree_map seems be not suspect but vmap does.
def combo_synchronize(params, hparams):
    params = jax.tree_map(lambda param: param + hparams[0], params)
    lr = jax.tree_map(lambda x: jnp.array(hparams[1], dtype=jnp.float32), params)    
    return params, lr


# Scale the set of learning rate and offset.
def scaling_sketch(mnmx, resolution):
    mn1, mx1, mn2, mx2 = mnmx
    gg1 = jnp.logspace(mn1, mx1, resolution)
    gg2 = jnp.logspace(mn2, mx2, resolution)
    lr0, lr1 = jnp.meshgrid(gg2, gg1)
    lr = jnp.stack([lr0.ravel(), lr1.ravel()], axis=1)
    return lr


if __name__ == "__main__":

    from pprint import pprint

    resnet = ResNet(10, nn.relu, ResNetBlock)
    variables = initialize(resnet, 42, jnp.ones((32, 28, 28, 1)))
    # pprint(jax.tree_map(jnp.shape, variables))

    # Tile test
    resolution = 8
    variables = jax.tree_map(lambda x: jnp.tile(x, (resolution**2,)+(1,)*len(x.shape)), variables)
    pprint(jax.tree_map(jnp.shape, variables))

    # Training test
    print(jax.devices())
    x = jnp.ones((32, 28, 28, 1))
    y = jnp.ones((32, ), dtype=int)
    lr = scaling_sketch((-4, 0, -4, 0), resolution)
    lr = lr.reshape((lr.shape[0], lr.shape[-1]))
    variables['params'], lr = combo_synchronize(variables['params'], lr)

    # vp
    variables = jax.tree_map(lambda x: x.reshape((4, int(x.shape[0]/4),)+(x.shape[1:])), variables)
    lr = jax.tree_map(lambda x: x.reshape((4, int(x.shape[0]/4),)+(x.shape[1:])), lr)
    variables, (loss, logits) = update_fn(variables, x, y, lr)

    print("Done")