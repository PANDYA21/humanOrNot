import time
mlp_start_time = time.time()
from mlp import *
mlp_end_time = time.time()

lr_start_time = time.time()
from lr import *
lr_end_time = time.time()

print('total MLP time: ', mlp_end_time - mlp_start_time)
print('total LR time: ', lr_end_time - lr_start_time)
