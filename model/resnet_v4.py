import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

import math
from tqdm import tqdm
from functools import partial
import optax
from tqdm import tqdm
from functools import partial


# resnet_kernel_init = jax.nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')
resnet_kernel_init = jax.nn.initializers.he_normal()


# FLAX construction
class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, on_train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not on_train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not on_train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init, use_bias=False)(x)
            x = nn.BatchNorm()(x, use_running_average=not on_train)
        x_out = self.act_fn(z + x)
        return x_out

class ResNet(nn.Module):
    num_classes : int
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, on_train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not on_train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, on_train=on_train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_classes, use_bias=False)(x)
        return x


# JAX run
def net(variables, x: jnp.array, on_train=True):
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # input.T
    x = jnp.transpose(x, [0, 3, 1, 2])

    # 1st conv
    x = jax.lax.conv(x, params['Conv_0']['kernel'], window_strides=(1, 1), padding='SAME')
    x, batch_stats['BatchNorm_0'] = batchnorm(x, params['BatchNorm_0'], batch_stats['BatchNorm_0'], on_train=on_train)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')

    # ResNetBlocks; conv0-conv1-skip
    for k, v in params.items():
        if 'ResNetBlock' in k:

            residual = x

            k_conv0 = v['Conv_0']['kernel']
            x = jax.lax.conv(x, k_conv0, window_strides=(1, 1), padding='SAME')
            x, batch_stats[k]['BatchNorm_0'] = batchnorm(x, v['BatchNorm_0'], batch_stats[k]['BatchNorm_0'], on_train=on_train)
            x = nn.relu(x)
            
            k_conv1 = v['Conv_1']['kernel']
            x = jax.lax.conv(x, k_conv1, window_strides=(1, 1), padding='SAME')
            x, batch_stats[k]['BatchNorm_1'] = batchnorm(x, v['BatchNorm_1'], batch_stats[k]['BatchNorm_1'], on_train=on_train)
            
            if 'Conv_2' in v.keys():
                k_conv2 = v['Conv_2']['kernel']
                residual = jax.lax.conv(residual, k_conv2, window_strides=(1, 1), padding='SAME')
            x += residual
            x = nn.relu(x)

    # FC
    x = nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    x = jnp.transpose(x, [0, 2, 3, 1])
    x = x.reshape((x.shape[0], -1))
    x = jnp.dot(x, params['Dense_0']['kernel'])

    # batch_stats
    variables = {'params': params, 'batch_stats': batch_stats}

    return nn.softmax(x), variables

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
    
@partial(jax.jit, static_argnums=3)
def loss_fn(variables, x, y, on_train=True):
    logits, variables = net(variables, x, on_train=on_train)
    return optax.softmax_cross_entropy_with_integer_labels(jnp.clip(logits, 1e-10, 1.), y).mean(), (logits, variables)

# @jax.jit
@partial(jax.vmap, in_axes=(0, None, None, 0))
@partial(jax.pmap, axis_name='batch', in_axes=(None, 0, 0, None), out_axes=(None, 0))
def update_fn(variables, x, y, lr):
    (loss, (logits, variables)), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables, x, y)
    # grads = jax.lax.pmean(grads, axis_name='batch')
    # jax.debug.print("params={v}", v=jax.tree_map(jnp.shape, variables['params']))
    # jax.debug.print("lr={v}", v=jax.tree_map(jnp.shape, lr))
    # jax.debug.print("grads mean={v}", v=grads['params']['ResNetBlock_0']['Conv_0']['kernel'].mean())
    # jax.debug.print("grads var={v}", v=grads['params']['ResNetBlock_0']['Conv_0']['kernel'].var())
    
    
    variables['params'] = jax.tree_map(lambda param, lr, g: param - lr * g, variables['params'], lr, grads['params'])
    return variables, (loss, logits)

# @partial(jax.vmap, in_axes=(0, None, None, 0, None))
def train_on_the_track(variables, batches, tbatches, hparams, epochs, desc=None):
    
    loss_archive, acc_archive = [], []
    tloss_archive, tacc_archive = [], []

    params, lr = duplicate_theta(variables['params'], hparams)
    variables['params'] = params

    for _ in tqdm(range(epochs), total=epochs, desc=desc, leave=False):
        loss, acc, tloss, tacc = train_and_validate_oneEpoch(variables, batches, tbatches, lr)
        loss_archive.append(loss)
        acc_archive.append(acc)
        tloss_archive.append(tloss)
        tacc_archive.append(tacc)
    
    # loss_archive = means_device(loss_archive, pmap_dim=1)
    # acc_archive = means_device(acc_archive, pmap_dim=1)
    # tloss_archive = means_device(tloss_archive, pmap_dim=1)
    # tacc_archive = means_device(tacc_archive, pmap_dim=1)
    
    return loss_archive, acc_archive, tloss_archive, tacc_archive

@partial(jax.vmap, in_axes=(0, 0), out_axes=(0, 0))     # induced trace; makes error; tree_map seems be not suspect but vmap does.
def duplicate_theta(params, hparams):
    params = jax.tree_map(lambda param: param + hparams[0], params)
    lr = jax.tree_map(lambda x: jnp.array(hparams[1], dtype=jnp.float32), params)    
    return params, lr

# @partial(jax.vmap, in_axes=(0, None, None, 0), out_axes=(0, 0, 0, 0))
def train_and_validate_oneEpoch(variables, batches, tbatches, lr):
    
    # training
    for batch in batches.as_numpy_iterator():
        # x = shard_data(batch['image'], 4)
        # y = shard_data(batch['label'], 4)
        x = batch['image']
        y = batch['label']
        variables, (loss, logits) = update_fn(variables, x, y, lr)
        # jax.debug.print("logits: {l}", l=logits)
        logits = logits.reshape((-1, logits.shape[-1]))
        y = y.reshape((-1,))
        acc = (logits.argmax(axis=-1)==y).mean()
        
    # validating
    # pmapped_loss_fn = jax.pmap(loss_fn, axis_name='batch', in_axes=(None, 0, 0, None))
    for tbatch in tbatches.as_numpy_iterator():
        # tx = shard_data(tbatch['image'], 4)
        # ty = shard_data(tbatch['label'], 4)
        tx = tbatch['image']
        ty = tbatch['label']
        
        # tloss, (tlogits, _) = pmapped_loss_fn(variables, tx, ty, False)
        tloss, (tlogits, _) = loss_fn(variables, tx, ty, False)
        tlogits = tlogits.reshape((-1, tlogits.shape[-1]))
        ty = ty.reshape((-1,))
        tacc = (tlogits.argmax(axis=-1)==y).mean()
        
    return loss, acc, tloss, tacc


def accuracy(logits, y):
    return (logits.argmax(axis=-1) == y)


def initialize(module, rng, x):
    variables = module.init(jax.random.PRNGKey(rng), x)
    variables['params']['Dense_0']['kernel'] = jax.nn.initializers.xavier_normal()(jax.random.PRNGKey(1), (50176, 10))  # 64 * 28**2
    # variables['params'] = jax.tree_map(lambda param: jnp.transpose(param, (3, 2, 0, 1)), variables['params'])
    def conv_dog(kp, x):
        kp = jax.tree_util.keystr(kp)
        if 'Conv' in kp:
            # x = x * 1e-6
            x = jnp.transpose(x, (3, 2, 0, 1))
        return x
    variables['params'] = jax.tree_util.tree_map_with_path(conv_dog, variables['params'])
    variables['batch_stats'] = jax.tree_map(lambda stats: stats.reshape((1, stats.shape[0], 1, 1)), variables['batch_stats'])
    return variables
        

if __name__ == "__main__":
    resolution = 3
    resnet11 = ResNet(num_classes=10, act_fn=nn.relu, block_class=ResNetBlock)
    variables = initialize(resnet11, 42, jnp.ones((1, 28, 28, 1)))
    variables = jax.tree_map(lambda x: jnp.tile(x, (resolution,)+(1,)*len(x.shape)), variables)
    # loss_archive, logit_archive, tloss_archive, tlogits_archive = train_on_the_track(variables, train_ds, lr=0.001, epochs=1000)
    print("Done!")