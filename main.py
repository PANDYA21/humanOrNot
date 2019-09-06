import time

# time the import of each module, to calculate total time
mlp_start_time = time.time()
from mlp import *
mlp_end_time = time.time()

mlp2_start_time = time.time()
from mlp2 import *
mlp2_end_time = time.time()

lr_start_time = time.time()
from lr import *
lr_end_time = time.time()

# print the times
print('total MLP (logistic) time: ', mlp_end_time - mlp_start_time)
print('total MLP (tanh) time: ', mlp2_end_time - mlp2_start_time)
print('total LR time: ', lr_end_time - lr_start_time)
