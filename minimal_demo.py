import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

import pickle

class Net(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28*28, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 10),
        ) 
        
    def forward(self, x):
        return self.net(x);

def main(**kwargs):
    train_size = kwargs['train_size']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    num_epochs = kwargs['num_epochs']
    hidden_dim = kwargs['hidden_dim']


    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    ds = MNIST('./dataset',train=False, download=True)
    
    data = ds.data.flatten(-2)
    data = data / data.max()
    train_data = [*zip(data[:train_size], ds.targets[:train_size], range(train_size))]

    sp = DistributedSampler(train_data, shuffle=False)
    print(f'rank[{rank}/{world_size-1}] sampler: {sp.num_replicas} {sp.num_samples} {sp.total_size}.')

    # Dataloader will be created for each node, the actual batch size for each node should be divided.
    actual_batch_size = batch_size // world_size
    dl = DataLoader(train_data, batch_size=actual_batch_size, sampler=sp, num_workers=2)

    net_without_ddp = Net(hidden_dim).to(rank)
    net = nn.parallel.DistributedDataParallel(net_without_ddp, device_ids=[rank])

    # since the loss is averaged across samples in a batch, and the gradients are averaged across nodes, the learning rate remains unchanged
    opt = Adam(net.parameters(), lr=learning_rate)
    
    if rank==0:
        log_dict = dict(test_acc=[], grad_scale=[])
    
    for ep in range(num_epochs):
        sp.set_epoch(ep) # set sampler epoch, it affects the randomness.
        net.train()
        for i, (batch, target, index) in enumerate(dl):
            batch = batch.to(rank).float()
            target = target.to(rank)

            out = net(batch)

            loss = F.cross_entropy(out, target)
            opt.zero_grad()
            loss.backward()
            if i==0:
                grad_scale = calc_gradient_scale(net_without_ddp)
                print(f'rank[{rank}/{world_size-1}] ep[{ep}/{num_epochs}] step: {i} actual_batch_size: {batch.shape[0]} grad_scale: {grad_scale:.4}')
            opt.step()
        
        net.eval()
        test_out = net(data[train_size:].to(rank))
        pred = test_out.argmax(-1)
        acc = torch.count_nonzero(pred == ds.targets[train_size:].to(rank)) / len(pred)

        if rank==0:
            log_dict['test_acc'].append(acc.cpu().item())
            log_dict['grad_scale'].append(grad_scale.cpu().item())

        print(f'rank[{rank}/{world_size-1}] ep[{ep}/{num_epochs}] loss: {loss/2:.4f} testacc: {acc:.4f}')
    if rank == 0:
        with open(f'./distsp_{world_size}gpu.pkl', 'wb') as f:
            pickle.dump(log_dict, f, pickle.HIGHEST_PROTOCOL)

def calc_gradient_scale(model):
    norms = (p.grad.detach().norm() for p in model.parameters())
    return sum(norms)

if __name__ == '__main__':
    args = {
        'train_size': 8000,
        'batch_size': 500,
        'learning_rate': 4e-4,
        'num_epochs': 100,
        'hidden_dim': 256,
    }

    main(**args)
