import pickle
import matplotlib.pyplot as plt

with open('./distsp_1gpu.pkl', 'rb') as f:
    gpu1 = pickle.load(f)

with open('./distsp_2gpu.pkl', 'rb') as f:
    gpu2 = pickle.load(f)

l2 = plt.plot(range(len(gpu2['test_acc'])), gpu2['test_acc'], label='distsp_2gpus')
l1 = plt.plot(range(len(gpu1['test_acc'])), gpu1['test_acc'], label='distsp_1gpu')
plt.legend()
plt.savefig('./test_acc.png')

plt.clf()
l2 = plt.plot(range(len(gpu2['grad_scale'])), gpu2['grad_scale'], label='distsp_2gpus')
l1 = plt.plot(range(len(gpu1['grad_scale'])), gpu1['grad_scale'], label='distsp_1gpu')
plt.legend()
plt.savefig('./grad_scale.png')