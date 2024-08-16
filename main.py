import argparse

# Load the settings
parser = argparse.ArgumentParser(description='원하는 하이퍼파라미터를 설정할 수 있습니다. eg. python main.py --resolution 256 --num_epochs 20 | 도움말. python main.py -h')
parser.add_argument('--resolution', default=8, type=int, help="mnmx에서 설정한 범위(learning rate and weight offest)를 얼마나 세세하게 관찰할지 결정할 수 있습니다.\n이 값의 제곱만큼 모델 수가 결정됩니다. 너무 작으면 fractal dimension이 구해지지 않을 수 있어요!")
parser.add_argument('--num_epochs', default=100, type=int, help="학습 시 epochs를 설정합니다. 20epochs면 거의 수렴해요.")
parser.add_argument('--batch_size', default=32, type=int, help="batch size를 결정합니다.")
parser.add_argument('--target_dim', default=10, type=int, help="이미지의 라벨종류가 몇개인가요?")
parser.add_argument('--nonlinearity', default='relu', type=str, help="convoluion 직후에 적용할 activation입니다. 지금은 'relu'와 'leaky'만 지원해요.")
parser.add_argument('--optimizer', default='sgd', type=str, help="optmizer입니다. 지금은 'sgd'와 'adam'만 지원해요. ")
parser.add_argument('--c_hidden', default=[16, 32, 64], type=int, nargs='+', help="block을 통과하고 나서 나오는 텐서의 채널 수를 설정합니다.\neg. --c_hidden 16 32 64")
parser.add_argument('--tile_batch', default=32, type=int, help="1번 training할 동안 몇 개의 모델을 동시에 굴릴지 설정합니다.\n모델 수(resolution**2)가 이거보다 많으면 절반으로 쪼개어 이거보다 작을 때까지 반복합니다.\n이 값이 클수록 한번에 많은 모델을 학습시킬 수 있지만, 너무 크면 OOM이 발생할 수 있습니다.\n2의 제곱수로 조절하세요.")
parser.add_argument('--mnmx', default=[-4, 0, -4, 0], type=int, nargs='+', help="learning rate의 범위 하한값, 상한값, weight offset의 범위 하한값, 상한값을 설정합니다. eg. --mnmx -4 0 -4 0")
parser.add_argument('--dpi', default=100, type=int, help="PNG 파일의 해상도 값입니다. 냅두셔도 됩니다.")
parser.add_argument('--figsize', default=[8, 8], type=int, nargs='+', help="lossmap의 figure size를 결정합니다. 정사각형꼴로 설정하세요.")
args = parser.parse_args()


from load_libraries import *
from protocol_save import *
from protocol_train import *
from protocol_plot import *
from datasets.mnist import *
from model.resnet_v4 import *
from typing import Any
from functools import partial
from pprint import pformat
import pandas as pd


# Metadata: hyperparmas
pd.DataFrame(args._get_kwargs()).to_csv(output_path + '/hyperparams.csv', header=False, index=False)

# Allocate the args
for arg in args._get_kwargs():
    k, v = arg
    if not isinstance(v, str):
        exec('%s = %s' % (k, v))
    else:
        exec('%s = "%s"' % (k, v))


# Define activations and optimizers
if nonlinearity == 'relu':
    exec('nonlinearity = nn.relu')
elif nonlinearity == 'leaky':
    exec('nonlinearity = nn.leaky_relu')

if optimizer == 'sgd':
    optimizer = optax.sgd
elif optimizer == 'adam':
    optimizer = optax.adam


def sketch_convmap(conv, title, saveas=None):
    plot_img(conv.reshape((resolution, resolution)), mnmx, title=title, savename=saveas)



# Scaling sketch
lrs = scaling_sketch(mnmx, resolution)

# Prepare dataset
train_ds, test_ds = prepare_dataset(batch_size)
total_batch = train_ds.cardinality().numpy()
total_tbatch = test_ds.cardinality().numpy()

batch = next(iter(train_ds))    # a batch for initializing
x, y = batch['image'], batch['label']

# Model loading
resnet20 = ResNet(10, nonlinearity, ResNetBlock)
variables = resnet20.init(jax.random.PRNGKey(1), x)     ### 아래 tile까는 과정에서 batch_stats들이 params-like shape으로 바뀌어버림. 이거 어디 잘못됨.

# Session
msg_start = 'Training start!\n\n' + \
    'DEVICE: ' + f'{len(jax.devices())} ' + jax.lib.xla_bridge.get_backend().platform + '\n' + \
    'DIR: ' + output_path + '\n' + \
    'Hyperparams: \n' + '='*50 + '\n' + \
    pformat(args) + \
    '\n' + '='*50
send_alaram(msg_start)
# metrics_tile = train_step_tile(variables, lrs, resnet20, tile_batch=tile_batch)  # (px, epoch)
V, tV = split_and_train(
        resnet=resnet20,
        hparams=lrs,
        batches=train_ds,
        tbatches=test_ds,
        num_epochs=num_epochs,
        tile_batch=tile_batch
        )

# Draw fractal image
send_alaram('Drawing...')
for i in tqdm(range(1, num_epochs), total=num_epochs-1, desc='Drawing', leave=False):

    # acc_conv = convergence_measure(metrics_tile['accuracy'][:, :i+1])
    acc_conv = accuracy_measure(metrics_tile['accuracy'][:, :i+1])
    loss_conv = convergence_measure(metrics_tile['loss'][:, :i+1])
    
    # Measure FD and plot the bordelines and convolution maps
    acc_edges = get_edges(acc_conv, resolution)
    loss_edges = get_edges(loss_conv, resolution)

    acc_fd = estimate_fractal_dimension(acc_edges, output_path + f'/train/accuracy/fd/epoch{i:03d}.png')
    loss_fd = estimate_fractal_dimension(loss_edges, output_path + f'/train/loss/fd/epoch{i:03d}.png')

    # dessin_borderline(acc_edges, mnmx, title=f'Training-accuracy\n({i}epoch(s), FD={acc_fd})', saveas=output_path + f'/train/accuracy/border/epoch{i:03d}.png')
    dessin_borderline(loss_edges, mnmx, title=f'Training-loss\n({i}epoch(s), FD={loss_fd})', saveas=output_path + f'/train/loss/border/epoch{i:03d}.png')

    # draw convergence map
    sketch_convmap(acc_conv, title=f'Training-accuracy\n({i}epoch(s), FD={acc_fd})', saveas=output_path + f'/train/accuracy/epoch{i:03d}.png')
    sketch_convmap(loss_conv, title=f'Training-loss\n({i}epoch(s), FD={loss_fd})', saveas=output_path + f'/train/loss/epoch{i:03d}.png')


# Animation as epochs
animate_sketches(output_path + '/train/loss')
animate_sketches(output_path + '/train/loss/border')
animate_sketches(output_path + '/train/accuracy')
animate_sketches(output_path + '/train/accuracy/border')

send_alaram('Successfully over!\nDIR: ' + output_path)


