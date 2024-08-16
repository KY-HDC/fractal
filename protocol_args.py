import argparse
import pandas as pd
from protocol_save import *

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

# Allocate the args
for arg in args._get_kwargs():
    k, v = arg
    if not isinstance(v, str):
        exec('%s = %s' % (k, v))
    else:
        exec('%s = "%s"' % (k, v))

# Metadata recording: hyperparams info
pd.DataFrame(args._get_kwargs()).to_csv(output_path + '/hyperparams.csv', header=False, index=False)