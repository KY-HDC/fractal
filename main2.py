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
import pandas as pd
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the settings
# parser = argparse.ArgumentParser(description='You can choose hyperparams which you want.')
# parser.add_argument('--target_dim', default=10, help="Number of kind as image's label.")
# parser.add_argument('--num_epochs', default=30, help="Training epochs.")
# parser.add_argument('--nonlinearity', default='relu', help="Activation function after convolution. Support as 'relu' or 'leaky'.")
# parser.add_argument('--c_hidden', default=[16, 32, 64], help="Output of channels after the block.")
# parser.add_argument('--batch_size', default=32, help="Divide the dataset by batch size.")
# parser.add_argument('--tile_batch', default=8, help="Set the amount of models during 1 learning session. If occuring OOM, double this argument. eg. 2, 4, 8, ...")
# parser.add_argument('--resolution', default=4, help="Divide the mnmx by resolution. If too small, fractal dimension could not be measured. I recommend over than 16.")
# parser.add_argument('--mnmx', default=[-4, 0, -4, 0], help="[min of lr, max of lr, min of W's offset, max of W's offset]")
# parser.add_argument('--dpi', default=100, help="PNG file's dpi.")
# parser.add_argument('--figsize', default=[8, 8], help="You can fix the lossmap's figure size.")
# parser.add_argument('--optimizer', default='sgd', help="Support as 'sgd' or 'adam'.")

parser = argparse.ArgumentParser(description='원하는 하이퍼파라미터를 설정할 수 있습니다.')
parser.add_argument('--resolution', default=4, type=int, help="mnmx에서 설정한 범위(learning rate and weight offest)를 얼마나 세세하게 관찰할지 결정할 수 있습니다.\n이 값의 제곱만큼 모델 수가 결정됩니다. 너무 작으면 fractal dimension이 구해지지 않을 수 있어요!")
parser.add_argument('--num_epochs', default=30, type=int, help="학습 시 epochs를 설정합니다. 20epochs면 거의 수렴해요.")
parser.add_argument('--batch_size', default=32, type=int, help="batch size를 결정합니다.")
parser.add_argument('--target_dim', default=10, type=int, help="이미지의 라벨종류가 몇개인가요?")
parser.add_argument('--nonlinearity', default='relu', type=str, help="convoluion 직후에 적용할 activation입니다. 지금은 'relu'와 'leaky'만 지원해요.")
parser.add_argument('--optimizer', default='sgd', type=str, help="optmizer입니다. 지금은 'sgd'와 'adam'만 지원해요. ")
parser.add_argument('--c_hidden', default=[16, 32, 64], type=int, nargs='+', help="block을 통과하고 나서 나오는 텐서의 채널 수를 설정합니다. 3개의 채널 수를 리스트로 표현해주세요.")
parser.add_argument('--tile_batch', default=8, type=int, help="1번 training할 동안 몇 개의 모델을 동시에 굴릴지 설정합니다.\n모델 수(resolution**2)가 이거보다 많으면 절반으로 쪼개어 이거보다 작을 때까지 반복합니다.\n이 값이 클수록 한번에 많은 모델을 학습시킬 수 있지만, 너무 크면 OOM이 발생할 수 있습니다.\n2의 제곱수로 조절하세요.")
parser.add_argument('--mnmx', default=[-4, 0, -4, 0], type=int, nargs='+', help="[learning rate의 범위 하한값, learning rate의 범위 상한값, weight offset의 범위 하한값, weight offset의 범위 상한값]")
parser.add_argument('--dpi', default=100, type=int, help="PNG 파일의 해상도 값입니다. 냅두셔도 됩니다.")
parser.add_argument('--figsize', default=[8, 8], type=int, nargs='+', help="lossmap의 figure size를 결정합니다. 정사각형꼴로 설정하세요.")
args = parser.parse_args()

# Metadata: hyperparmas
pd.DataFrame(args._get_kwargs()).to_csv(output_path + '/hyperparams.csv', header=False, index=False)

# Define args
# hyperparams = {}
for arg in args._get_kwargs():
    k, v = arg
    if not isinstance(v, str):
        exec('%s = %s' % (k, v))
    else:
        exec('%s = "%s"' % (k, v))

# Generate the metadata    
# with open(output_path + '/hyperparams.json', 'w') as js:
#     json.dump(arg, js, indent=4)

# Define activations and optimizers
if nonlinearity == 'relu':
    exec('nonlinearity = nn.relu')
elif nonlinearity == 'leaky':
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
    for _ in tqdm(range(num_epochs), total=num_epochs, leave=False, desc='Epochs'):
        for batch in tqdm(train_ds.as_numpy_iterator(), total=total_batch, leave=False, desc='Iter'):
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
    pformat(args) + \
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
