import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import conv

State = jnp

class ResNetBlock:
    def __init__(self, state: State):
        ...