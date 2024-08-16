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

V, tV = split_and_train(
        resnet=resnet20,
        hparams=lrs,
        batches=train_ds,
        tbatches=test_ds,
        num_epochs=num_epochs,
        tile_batch=tile_batch
        )
