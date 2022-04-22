# Minimal Example For DDP in Pytorch

## Experiments

1. Using MNIST classification as a demo task, experimented the effects of `batch_size`, `learning_rate`, `num_epochs` in DataDistributedParallel.
2. Add `DistributedSampler` to better eliminate difference between different numbers of gpu devices.

## Conclusions

`DataLoader` will be initialized for every gpu, with different shuffles. For one step of optimizer, the gradients are gathered across all devices, and `batch_size` and `lr` are not halved for each gpu.

To get the same training curve of 2 gpus using only 1 gpu, `num_epochs` has to be doubled, and update parameters for every other step with two steps' average loss.

With `DistributedSampler`, the best practice is to evenly divide batch size by number of gpus.

## To Run This

Run `CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port 9701 --nproc_per_node=2 minimal_demo.py` in Repo directory.

