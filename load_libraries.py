# GPU settings
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4'    # multi
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'          # single

# Computation library
import jax
from jax import lax, random, config, numpy as jnp
# config.update('jax_enable_x64', True)   # for double precision, but cannot run convolution!
from flax.training import train_state
import porespy as ps
import numpy as np

# Plotting settings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib import cm

# ETC
from functools import partial
from typing import Callable, Any
from pprint import pprint

from tqdm import tqdm
from collections import deque

import json

class TrainState(train_state.TrainState):
    batch_stats: Any
