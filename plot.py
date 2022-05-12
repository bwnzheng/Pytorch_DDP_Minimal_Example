import numpy as np
import matplotlib.pyplot as plt


gpu2 = np.load('./distsp_gpu2.npy')
gpu1 = np.load('./distsp_gpu1.npy')
# gpu1 = gpu1[0:len(gpu1):2]
l2 = plt.plot(range(len(gpu2)), gpu2, label='distsp_gpu2')
l1 = plt.plot(range(len(gpu1)), gpu1, label='distsp_gpu1')
plt.legend()
plt.savefig('./2.png')