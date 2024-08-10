import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import numpy as np
from flax.training import train_state
from flax import jax_utils
from functools import partial

import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import logging
from tqdm import tqdm

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


@partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(rng, learning_rate, momentum):
  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)


@partial(jax.pmap, axis_name='ensemble')
def apply_model(state, images, labels):
  def loss_fn(params):
    logits = CNN().apply({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  probs = jax.lax.pmean(jax.nn.softmax(logits), axis_name='ensemble')
  accuracy = jnp.mean(jnp.argmax(probs, -1) == labels)
  return grads, loss, accuracy

@jax.pmap
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = jax_utils.replicate(train_ds['image'][perm, ...])
    batch_labels = jax_utils.replicate(train_ds['label'][perm, ...])
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(jax_utils.unreplicate(loss))
    epoch_accuracy.append(jax_utils.unreplicate(accuracy))
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy

def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  return train_ds, test_ds


learning_rate = 0.001
momentum = 0.9
num_epochs = 100
batch_size = 32

train_ds, test_ds = get_datasets()
test_ds = jax_utils.replicate(test_ds)
rng = jax.random.key(0)

rng, init_rng = jax.random.split(rng)
state = create_train_state(jax.random.split(init_rng, jax.device_count()),
                           learning_rate, momentum)

for epoch in tqdm(range(1, num_epochs + 1)):
  rng, input_rng = jax.random.split(rng)
  state, train_loss, train_accuracy = train_epoch(
      state, train_ds, batch_size, input_rng)

  _, test_loss, test_accuracy = jax_utils.unreplicate(
      apply_model(state, test_ds['image'], test_ds['label']))

  logging.info(
      'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, '
      'test_loss: %.4f, test_accuracy: %.2f'
      % (epoch, train_loss, train_accuracy * 100, test_loss,
         test_accuracy * 100))