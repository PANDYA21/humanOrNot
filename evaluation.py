import time
mlp_start_time = time.time()
from mlp import *
mlp_end_time = time.time()

lr_start_time = time.time()
from lr import *
lr_end_time = time.time()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# accuracy plot
axs[0].boxplot([accs_mlp, accs_lr])
axs[0].set_title('Accuracy')
# plt.show()

# F!-score plot
axs[1].boxplot([fs_mlp, fs_lr])
axs[1].set_title('F1-Sore')

# labels
plt.setp(
  axs, 
  xticks=[y + 1 for y in range(len([accs_mlp, accs_lr]))],
  xticklabels=['mlp', 'lr'])
plt.show()

print(mlp_end_time - mlp_start_time)
print(lr_end_time - lr_start_time)
