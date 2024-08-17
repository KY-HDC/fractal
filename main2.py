from protocol_args import (
    resolution, num_epochs, batch_size, target_dim, nonlinearity, optimizer,
    c_hidden, tile_batch, mnmx, dpi, figsize
)
from load_libraries import *
from protocol_save import *
from protocol_train import *
from protocol_plot import *
from datasets.mnist import train_ds, test_ds, info, x, y
from typing import Any
from functools import partial
from pprint import pformat
import pandas as pd

from model.flax_resnet import *
from model.jax_resnet import *

# Scaling sketch
lrs = scaling_sketch(mnmx, resolution)

# Model loading
actfn = nn.relu if nonlinearity=="relu" else nn.leaky_relu
resnet20 = ResNet(10, actfn, ResNetBlock)
variables = resnet20.init(jax.random.PRNGKey(1), x)     ### 아래 tile까는 과정에서 batch_stats들이 params-like shape으로 바뀌어버림. 이거 어디 잘못됨.


'''240818
아래 function을 다시 써야되나 싶다.
메모리 관리랑 kernel들 shape 관리 차원에서 너무 애로사항이 꽃핀다.
split하는건 꽤 괜찮은 생각이었지만 메모리가 얼마 안남은 경우 이 마저도 도움이 안된다.
그나저나 python에선 method가 한번 실행되고 있으면 메모리를 얼마나 먹는걸까?

'''


V, tV = split_and_train(
        resnet=resnet20,
        hparams=lrs,
        batches=train_ds,
        tbatches=test_ds,
        num_epochs=num_epochs,
        tile_batch=tile_batch
        )

print(v.shape, tV.shape)