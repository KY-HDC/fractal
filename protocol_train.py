import jax
import optax
import jax.numpy as jnp
from model.flax_resnet import *
from model.jax_resnet import *

from functools import partial
from tqdm import tqdm

from datasets.mnist import x, y

x_init, y_init = x, y

def split_and_train(resnet, hparams, batches, tbatches, num_epochs, tile_batch):
    '''
    240818. 
    첫 tile의 학습 이후, 두번째 tile에서 kernel shape가 이상하게 되어 들어온다.
    그리고 메모리 관리에서도 문제가 생긴다. 메모리 정리가 덜됐을 때 다시 돌린 경우,
    tile_batch=10도 버티지못하고 6에서 끊긴다.
    그러므로 이 문제를 해결하려면 tile 각각 loss를 저장해서 메모리는 없애는 식으로
    해야될 것 같다. 한번에 돌린답시고 kernel 등을 기억해야 되다보니
    메모리가 과다하게 사용되는 것 같다.
    split 후 initialize->tiling->train이 이루어지는데 tiling은 
    '''
    # Cut off the learning rates as bite size
    bs = hparams.shape[0]
    print("bs(raw):", bs)
    if bs > tile_batch:
        train_loss1, test_loss1 = split_and_train(resnet, hparams[:bs//2, ], batches, tbatches, num_epochs, tile_batch)
        train_loss2, test_loss2 = split_and_train(resnet, hparams[bs//2:, ], batches, tbatches, num_epochs, tile_batch)
        return jnp.concatenate((train_loss1, train_loss2), axis=0), jnp.concatenate((test_loss1, test_loss2), axis=0)

    print("bs", bs)
    variables = initialize(resnet, 42, x_init)  # kernel's shape=(OIHW)
    variables = jax.tree_map(lambda v: jnp.tile(v, (bs,)+(1,)*len(v.shape)), variables)    # kernel's shape=(rOIHW); r=(P)V=resolution**2
    hparams = hparams.reshape((hparams.shape[0], hparams.shape[-1]))
    variables['params'], hparams = combo_synchronize(variables['params'], hparams)
    print("After split, ", jax.tree_map(jnp.shape, variables['params']))
    ### tree_map
    # apply each other offset and learning rate
    # desc = f'[Tile {tile_batch}/{math.ceil(resolution**2/tile_batch)}] Training-epochs: '
    # print(jax.tree_map(jnp.shape, variables))
    loss_archive, acc_archive, tloss_archive, tacc_archive = train_on_the_track(
        variables, batches, tbatches, hparams, num_epochs)

    # Find the well-converged train loss and get them up!
    loss_arr = np.array(loss_archive).T
    acc_arr = np.array(acc_archive).T
    tloss_arr = np.array(tloss_archive).T
    tacc_arr = np.array(tacc_archive).T

    return convergence_measure(jnp.stack(loss_archive, axis=-1)), convergence_measure(jnp.stack(tloss_archive, axis=-1))    # stack-->(16, 100), convergence_measure-->(...?)

@partial(jax.vmap, in_axes=(0,), out_axes=0)
def convergence_measure(v, max_val=1e6):
    fin = jnp.isfinite(v)
    v = v * fin + max_val * (1-fin)
    v /= v[0]
    exceeds = (v > max_val)
    v = v * (1-exceeds) + max_val * exceeds
    # converged = (jnp.mean(v[-20:]) < 1)
    return -(1-jnp.mean(v))


def train_on_the_track(variables, batches, tbatches, hparams, epochs):
    ''' Variables' kernel: `((P)VHWIO)`; resolution**2=PV \\
        Batches' image: `(BHWC)`; N=B=batch_size \\
        Hparams: `((P)V, 2)`
    '''

    loss_archive, acc_archive = [], []
    tloss_archive, tacc_archive = [], []

    # params, lr = duplicate_theta(variables['params'], hparams)
    # variables['params'] = params
    # print(jax.tree_map(jnp.shape, hparams))
    for _ in tqdm(range(epochs), total=epochs, leave=False):
        
        loss, acc, tloss, tacc = train_and_validate_oneEpoch(variables, batches, tbatches, hparams)
        loss_archive.append(loss)
        acc_archive.append(acc)
        tloss_archive.append(tloss)
        tacc_archive.append(tacc)
    
    return loss_archive, acc_archive, tloss_archive, tacc_archive

# @partial(jax.vmap, in_axes=(0, 0), out_axes=(0, 0))     # induced trace; makes error; tree_map seems be not suspect but vmap does.
# def duplicate_theta(params, hparams):
#     params = jax.tree_map(lambda param: param + hparams[0], params)
#     lr = jax.tree_map(lambda x: jnp.array(hparams[1], dtype=jnp.float32), params)    
#     return params, lr


def train_and_validate_oneEpoch(variables, batches, tbatches, lr):
    ''' Variables' kernel: `((P)VHWIO)`; resolution**2=PV \\
    Batches' image: `(BHWC)`; N=B=batch_size \\
    lr(Hparams): `((P)V, 2)`
    '''

    # Training
    for batch in batches:
        x = batch['image']
        y = batch['label']
        # print("V=", str(jax.tree_map(jnp.shape, variables))[:200])
        # print("X=", str(jax.tree_map(jnp.shape, x))[:200])
        # print("LR=", str(jax.tree_map(jnp.shape, lr))[:200])
        # print("In train_and_validate_oneEpoch, X:", x.shape, "VS P:", variables['params']['Conv_0']['kernel'].shape)

        variables, (loss, logits) = update_fn(variables, x, y, lr)
        # print('loss', loss.shape)   # 
        # print('logits', logits.shape)
        # print('infer: ', (logits.argmax(axis=-1)==y).shape)
        # print('Y: ', y.shape)

        # logits = logits.reshape((-1, logits.shape[-1]))
        # y = y.reshape((-1,))
        # y = jnp.tile(y)
        acc = (logits.argmax(axis=-1)==y).mean(axis=-1)
        # print('acc: ', acc.shape)
        # acc = jnp.mean(acc, axis=1)
        
    # Evaluation
    for tbatch in tbatches:
        tx = tbatch['image']
        ty = tbatch['label']
        tloss, (tlogits, _) = loss_fn(variables, tx, ty, False)
        # tlogits = tlogits.reshape((-1, tlogits.shape[-1]))
        # ty = ty.reshape((-1,))
        tacc = (tlogits.argmax(axis=-1)==y).mean(axis=-1)
        
    return loss, acc, tloss, tacc


def accuracy(logits, y):
    return (logits.argmax(axis=-1) == y)


# # ResNet's training and evaluation
# def initialize(module, rng, x):
#     variables = module.init(jax.random.PRNGKey(rng), x)
#     variables['params']['Dense_0']['kernel'] = jax.nn.initializers.xavier_normal()(jax.random.PRNGKey(1), (50176, 10))  # 64 * 28**2
#     # variables['params'] = jax.tree_map(lambda param: jnp.transpose(param, (3, 2, 0, 1)), variables['params'])
#     def conv_dog(kp, x):
#         kp = jax.tree_util.keystr(kp)
#         if 'Conv' in kp:
#             # x = x * 1e-6
#             x = jnp.transpose(x, (3, 2, 0, 1))
#         return x
#     variables['params'] = jax.tree_util.tree_map_with_path(conv_dog, variables['params'])
#     variables['batch_stats'] = jax.tree_map(lambda stats: stats.reshape((1, stats.shape[0], 1, 1)), variables['batch_stats'])
#     return variables

# @partial(jax.vmap, in_axes=(0, 0), out_axes=(0, 0))     # induced trace; makes error; tree_map seems be not suspect but vmap does.
# def duplicate_theta(params, hparams):
#     params = jax.tree_map(lambda param: param + hparams[0], params)
#     lr = jax.tree_map(lambda x: jnp.array(hparams[1], dtype=jnp.float32), params)    
#     return params, lr

# def train_on_the_track(variables, batches, tbatches, hparams, epochs, desc=None):
    
#     loss_archive, acc_archive = [], []
#     tloss_archive, tacc_archive = [], []
#     params, lr = duplicate_theta(variables['params'], hparams)
#     variables['params'] = params

#     for _ in tqdm(range(epochs), total=epochs, desc=desc, leave=False):
#         loss, acc, tloss, tacc = train_and_validate_oneEpoch(variables, batches, tbatches, lr)
#         loss_archive.append(loss)
#         acc_archive.append(acc)
#         tloss_archive.append(tloss)
#         tacc_archive.append(tacc)
    
#     return loss_archive, acc_archive, tloss_archive, tacc_archive

# def train_and_validate_oneEpoch(variables, batches, tbatches, lr):
#     # Training session
#     for batch in batches.as_numpy_iterator():
#         x = batch['image']
#         y = batch['label']
#         variables, (loss, logits) = update_fn(variables, x, y, lr)
#         logits = logits.reshape((-1, logits.shape[-1]))
#         y = y.reshape((-1,))
#         acc = (logits.argmax(axis=-1)==y).mean()
        
#     # Evaluating session
#     # pmapped_loss_fn = jax.pmap(loss_fn, axis_name='batch', in_axes=(None, 0, 0, None))
#     for tbatch in tbatches.as_numpy_iterator():
#         # tx = shard_data(tbatch['image'], 4)
#         # ty = shard_data(tbatch['label'], 4)
#         tx = tbatch['image']
#         ty = tbatch['label']
        
#         # tloss, (tlogits, _) = pmapped_loss_fn(variables, tx, ty, False)
#         tloss, (tlogits, _) = loss_fn(variables, tx, ty, False)
#         tlogits = tlogits.reshape((-1, tlogits.shape[-1]))
#         ty = ty.reshape((-1,))
#         tacc = (tlogits.argmax(axis=-1)==y).mean()
        
#     return loss, acc, tloss, tacc

# # Model's internal session
# # TODO: How about optax?
# @partial(jax.vmap, in_axes=(0, None, None, 0))
# @partial(jax.pmap, axis_name='batch', in_axes=(None, 0, 0, None), out_axes=(None, 0))
# def update_fn(variables, x, y, lr):
#     (loss, (logits, variables)), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables, x, y)
#     variables['params'] = jax.tree_map(lambda param, lr, g: param - lr * g, variables['params'], lr, grads['params'])
#     return variables, (loss, logits)

# @jax.jit
# def train_step(state, batch):
#     """Train for a single step."""
#     def loss_fn(params):
#         logits, updates = state.apply_fn(
#             {'params': params, 'batch_stats': state.batch_stats},
#             x=batch['image'], on_train=True, mutable=['batch_stats'])
#         loss = optax.softmax_cross_entropy_with_integer_labels(
#         logits=logits, labels=batch['label']).mean()
#         return loss, (logits, updates)
#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (loss, (logits, updates)), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     state = state.replace(batch_stats=updates['batch_stats'])
#     metrics = {
#         'loss': loss,
#         'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
#     }
#     return state, metrics

# @jax.jit
# def evaluate_step(state, batch):
#     """Evaluate for a single step."""
#     def loss_fn(params):
#         logits = state.apply_fn(
#             {'params': params, 'batch_stats': state.batch_stats},
#             x=batch['image'], on_train=False)
#         loss = optax.softmax_cross_entropy_with_integer_labels(
#         logits=logits, labels=batch['label']).mean()
#         return loss, logits
#     loss, logits = loss_fn(state.params)
#     metrics = {
#         'loss': loss,
#         'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
#     }
#     return metrics


# if __name__ == "__main__":
#     import os
#     os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

#     from datasets.mnist import *
#     from model.resnet_v4 import *
#     from flax.training import train_state
#     from typing import Any
#     import jax.numpy as jnp
#     import optax
#     train_ds, test_ds = prepare_dataset(32)
#     batch = next(iter(train_ds))
#     batch['image'] = jnp.array(batch['image'], dtype=jnp.float32)
#     batch['label'] = jnp.array(batch['label'], dtype=jnp.int8)

#     # lowered HLO
#     class TrainState(train_state.TrainState):
#         batch_stats: Any

#     model = ResNet(10, flax.linen.relu, ResNetBlock)
#     variables = model.init(jax.random.PRNGKey(1), batch['image'])
#     state = TrainState.create(
#         apply_fn=model.apply,
#         params=variables['params'],
#         batch_stats=variables['batch_stats'],
#         tx=optax.sgd(learning_rate=0.001)
#         )
#     lowered = train_step.lower(state, batch)
#     print(lowered.as_text())

#     # compiled HLO
#     compiled = lowered.compile()
#     print(compiled.cost_analysis()[0]['flops'])
    
#     # Run
#     print(compiled(state, batch))

#     # Compile-check evaluate_step, too
#     lowered_eval = evaluate_step.lower(state, batch)
#     compiled_eval = lowered_eval.compile()
#     print(lowered_eval.as_text())
#     print(compiled_eval.cost_analysis()[0]['flops'])
    
#     # pmapping check
#     print(jax.devices())
#     @jax.pmap
#     def create_first(rng):
#         model = ResNet(10, flax.linen.relu, ResNetBlock)
#         params = model.init(rng, batch['image'])['params']
#         return TrainState.create(
#             apply_fn=model.apply,
#             params=params,
#             batch_stats=variables['batch_stats'],
#             tx=optax.sgd(learning_rate=0.001)
#         )
#     rng = jax.random.split(jax.random.PRNGKey(42), 4)
#     p_state = create_first(rng)
    
#     p_train = jax.pmap(
#         train_step, 
#         axis_name='i', 
#         in_axes=(0, None),
#         devices=jax.devices()
#         )(p_state, batch)
#     print(len(p_train))
#     print(jax.tree_map(jnp.shape, p_train))

#     # vp; 굳이 vmap이 필요한가? params random으로 쪼개려고 넣는데... 난 이거 필요없음
#     # @partial(jax.vmap, in_axes=0)
#     @jax.pmap
#     def create_first(rng):
#         model = ResNet(10, flax.linen.relu, ResNetBlock)
#         params = model.init(rng, batch['image'])['params']
#         return TrainState.create(
#             apply_fn=model.apply,
#             params=params,
#             batch_stats=variables['batch_stats'],
#             tx=optax.sgd(learning_rate=0.001)
#         )
#     rng = jax.random.split(jax.random.PRNGKey(42), 4)
#     vp_state = create_first(rng)
#     print(jax.tree_map(jnp.shape, vp_state))

    
#     @jax.pmap
#     def train_step(state, batch):
#         """Train for a single step."""
#         def loss_fn(params):
#             logits, updates = ResNet(10, flax.linen.relu, ResNetBlock).apply_fn(
#                 {'params': params, 'batch_stats': state.batch_stats},
#                 x=batch['image'], on_train=True, mutable=['batch_stats'])
#             loss = optax.softmax_cross_entropy_with_integer_labels(
#             logits=logits, labels=batch['label']).mean()
#             return loss, (logits, updates)
#         grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#         print('grading')
#         (loss, (logits, updates)), grads = grad_fn(state.params)
#         state = state.apply_gradients(grads=grads)
#         state = state.replace(batch_stats=updates['batch_stats'])
#         metrics = {
#             'loss': loss,
#             'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
#         }
#         return state, metrics
#     train_step(vp_state, batch)
#     logits, updates = ResNet(10, flax.linen.relu, ResNetBlock).apply(
#                 {'params': state.params, 'batch_stats': state.batch_stats},
#                 x=batch['image'], on_train=True, mutable=['batch_stats'])
#     # vp_train = jax.pmap(
#     #     train_step, 
#     #     axis_name='i', 
#     #     in_axes=(0, None),
#     #     devices=jax.devices()
#     #     )
#     # vp_train = jax.vmap(vp_train, in_axes=(1, None))(vp_state, batch)
#     # print(len(vp_train))

#     # # 다시 병렬의 길이 열린듯
#     # model = ResNet(10, flax.linen.relu, ResNetBlock)
#     # params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
    
#     # lr = jnp.array([0.001, 0.002, 0.003])
#     # lr = jnp.linspace(1, 20, 20) * 0.001
#     # lr = lr.reshape((4, 5))     # (4, 5, 1)
    
#     # def create_first(lr):
#     #     return TrainState.create(
#     #         apply_fn=model.apply,
#     #         params=params,
#     #         batch_stats=variables['batch_stats'],
#     #         tx=optax.sgd(learning_rate=lr)
#     #     )
    
#     # # v(p(f)) -> O if (4, 5)
#     # parallelized = jax.pmap(create_first, devices=jax.devices())
#     # jax.tree_map(jnp.shape, parallelized(lr))
#     # vectorized = jax.vmap(parallelized, in_axes=1)
#     # jax.tree_map(jnp.shape, vectorized(lr))

#     # state = vectorized(lr)
#     # TrainState.

#     # jax.tree_map(jnp.shape, batch)
#     # # batch = jax.tree_map(partial(jnp.reshape, ()))
#     # jax.pmap(lambda state, x: model.apply(state, x), in_axes=(1, None))(state, batch)
#     # jax.vmap(lambda state, x: model.apply(state, x), in_axes=(1, None))(state, batch)
#     # model.apply(state, batch)
#     # jax.vmap(jax.pmap(lambda state, x: model.apply(state, x), in_axes=(0, None)), in_axes=(1, None))(state, batch)
#     # jax.pmap(jax.vmap(lambda state, x: model.apply(state, x), in_axes=(0, None)), in_axes=(1, None))(state, batch)

#     # # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
    
    

#     # # p(v(f)) -> X if (4, 5)
#     # vectorized = jax.vmap(create_first, in_axes=1)
#     # jax.tree_map(jnp.shape, vectorized(lr))
#     # parallelized = jax.pmap(vectorized, in_axes=0)
#     # jax.tree_map(jnp.shape, parallelized(lr))
    
#     # jax.pmap(
#     #     train_step, 
#     #     axis_name='i', 
#     #     in_axes=(0, None),
#     #     devices=jax.devices()
#     #     )(parallelized(lr), batch)