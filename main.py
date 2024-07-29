import os
from load_libraries import *
from protocol_save import *
from protocol_train import *
from protocol_plot import *
from datasets.mnist import *
from model.resnet_v4 import *
from typing import Any
from functools import partial
from pprint import pprint, pformat
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the settings
with open('hyperparams.json', 'r') as js:
    hyperparams = json.load(js)
for k, v in hyperparams.items():
    if not isinstance(v, str):
        exec('%s = %s' % (k, v))
    else:
        exec('%s = "%s"' % (k, v))

# Copy as metadata    
with open(output_path + '/hyperparams.json', 'w') as js:
    json.dump(hyperparams, js, indent=4)


# Define activations and optimizers
if nonlinearity == 'relu':
    # nonliearity = nn.relu
    exec('nonlinearity = nn.relu')
elif nonlinearity == 'leaky':
    # nonlinearity = nn.leaky_relu
    exec('nonlinearity = nn.leaky_relu')


if optimizer == 'sgd':
    optimizer = optax.sgd
elif optimizer == 'adam':
    optimizer = optax.adam

# Scaling sketch
lrs = scaling_sketch(mnmx, resolution)

# Prepare dataset
train_ds, test_ds = prepare_dataset(batch_size)
total_batch = train_ds.cardinality().numpy()
total_tbatch = test_ds.cardinality().numpy()

for batch in train_ds.as_numpy_iterator():
    x = batch['image']
    y = batch['label']
    break

# Model loading
resnet20 = ResNet(10, nonlinearity, ResNetBlock)
variables = resnet20.init(jax.random.PRNGKey(1), x)

# Tiling and plotting functions
@partial(jax.vmap, in_axes=(None, 0, None))
def train_step_v(variables, lr, model):
    state = TrainState.create(
        apply_fn=model.apply,
        params=jax.tree_map(lambda param: param + lr[0], variables['params']),
        batch_stats=variables['batch_stats'],
        tx=optimizer(lr[1])
    )

    M_train = []
    for _ in tqdm(range(num_epochs), total=num_epochs):
        for batch in tqdm(train_ds.as_numpy_iterator(), leave=False, desc='Training'):
            state, metrics = train_step(state, batch)
        M_train.append(metrics)

    return M_train

def make_array(metrics_v, target):
    return np.vstack([metrics_v[i][target] for i in range(num_epochs)]).T     # (px, epochs)

def train_step_tile(variables, lrs, model, tile_batch=tile_batch):
    bs = lrs.shape[0]
    particles = bs//tile_batch
    if particles > 1:
        print(f"Splitting as {bs}->{bs//2}tiles.")
    if bs > tile_batch:
        metrics_v1 = train_step_tile(variables, lrs[:bs//2], model)
        metrics_v2 = train_step_tile(variables, lrs[bs//2:], model)

        acc_v1, loss_v1 = make_array(metrics_v1, 'accuracy'), make_array(metrics_v1, 'loss')
        acc_v2, loss_v2 = make_array(metrics_v2, 'accuracy'), make_array(metrics_v2, 'loss')

        acc = np.vstack([acc_v1, acc_v2])
        loss = np.vstack([loss_v1, loss_v2])

        return {'accuracy': acc, 'loss': loss}
    return train_step_v(variables, lrs, model)

def sketch_convmap(conv, title, saveas=None):
    plot_img(conv.reshape((resolution, resolution)), mnmx, title=title, savename=saveas)
    # print(f"Lossmap was drawn as '{saveas}'")


# Session
msg_start = 'Training start!\n\n' + \
    'DIR: ' + output_path + '\n' + \
    '<hyperparams>\n==================\n' + \
    pformat(hyperparams) + \
    '\n=================='
send_alaram(msg_start)
metrics_tile = train_step_tile(variables, lrs, resnet20, tile_batch=tile_batch)  # (px, epoch)

for i in range(num_epochs):
    if i == 0:
        acc_conv = convergence_measure(metrics_tile['accuracy'][:, [0]])
        loss_conv = convergence_measure(metrics_tile['loss'][:, [0]])
    else:
        acc_conv = convergence_measure(metrics_tile['accuracy'][:, :i])
        loss_conv = convergence_measure(metrics_tile['loss'][:, :i])
    
    # measure FD
    acc_fd = estimate_fractal_dimension(acc_conv, resolution, output_path + f'/train/accuracy/fd/epoch{i:03d}.png')
    loss_fd = estimate_fractal_dimension(loss_conv, resolution, output_path + f'/train/loss/fd/epoch{i:03d}.png')

    # draw convergence map
    sketch_convmap(acc_conv, title=f'Training-accuracy\n({i}epoch(s), FD={acc_fd})', saveas=output_path + f'/train/accuracy/epoch{i:03d}.png')
    sketch_convmap(loss_conv, title=f'Training-loss\n({i}epoch(s), FD={loss_fd})', saveas=output_path + f'/train/loss/epoch{i:03d}.png')

    if i == num_epochs//2:
        send_alaram('Almost half of epochs.')

send_alaram('Successfully over!\nDIR: ' + output_path)
