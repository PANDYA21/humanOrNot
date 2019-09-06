from mlp import *
import matplotlib.pyplot as plt

fig, axs = plt.subplots()

# basic plot
axs.boxplot([accs])
axs.set_title('basic plot')
plt.show()