import jax
import jax.numpy as jnp
from jax.lax import conv, conv_general_dilated
import numpy as np

# N, H, W, C = 10, 7, 7, 1
# I, O = C, 32
# kernel_size = (3, 3)
# strides = (1, 1)
# padding = 'SAME'

# x = jnp.ones((N, H, W, C))
# kernel = jnp.ones(kernel_size + (I, O))

# x = jnp.transpose(x, [0, 3, 1, 2])
# kernel = jnp.transpose(kernel, [3, 2, 0, 1])

# output = conv(x, kernel, window_strides=strides, padding=padding)
# print(output[1][0])


# class Counter:
#     def __init__(self):
#         self.n = 0

#     def count(self):
#         self.n += 1
#         return self.n
    
#     def reset(self):
#         self.n = 0

# counter = Counter()
# counter.reset()
# fast_count = jax.jit(counter.count)
# for _ in range(5):
#     print(fast_count())


CounterState = int

class CounterV2:
    def count(self, n: CounterState) -> tuple[int, CounterState]:
        return n+1, n+1
    
    def reset(self) -> CounterState:
        return 0
    
counter = CounterV2()
state = counter.reset()
fast_count = jax.jit(counter.count)
for _ in range(5):
    value, state = fast_count(state)
    print(value)