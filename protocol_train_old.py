import jax
import optax
import jax.numpy as jnp

def scaling_sketch(mnmx, resolution):
    mn1, mx1, mn2, mx2 = mnmx
    gg1 = jnp.logspace(mn1, mx1, resolution)
    gg2 = jnp.logspace(mn2, mx2, resolution)
    lr0, lr1 = jnp.meshgrid(gg2, gg1)
    lr = jnp.stack([lr0.ravel(), lr1.ravel()], axis=-1)
    return lr

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'], on_train=True, mutable=['batch_stats'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics

@jax.jit
def evaluate_step(state, batch):
    """Evaluate for a single step."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'], on_train=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
        return loss, logits
    loss, logits = loss_fn(state.params)
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return metrics


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