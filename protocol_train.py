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
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'], on_train=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
        return loss, (logits, updates)
    loss, (logits, _) = loss_fn(state.params)
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return metrics
