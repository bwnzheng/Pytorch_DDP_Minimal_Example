import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28*28,256),
            nn.Sigmoid(),
            nn.Linear(256,10),
            nn.Sigmoid(),
        ) 
        
    def forward(self, x):
        return self.net(x);



if __name__ == '__main__':
    train_size = 8000
    batch_size = 500
    learning_rate = 4e-4


    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size==2:
        num_epochs = 100
    elif world_size==1:
        num_epochs = 200
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    ds = MNIST('./dataset',train=False, download=True)
    
    data = ds.data.flatten(-2)
    data = data / data.max()

    dl = DataLoader([*zip(data[:train_size], ds.targets[:train_size])], batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net().to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    opt = Adam(net.parameters(), lr=learning_rate)
    

    test_acc_list = []
    
    for ep in range(num_epochs):
        net.train()
        for i, (batch, target) in enumerate(dl):
            if i==0:
                # print(f'rank[{rank}/{world_size-1}] ep[{ep}/{num_epochs}] step: {i} actual_batch_size: {batch.shape[0]} target: {target[:5]}')
                pass
            batch = batch.to(rank).float()
            target =  target.to(rank)

            out = net(batch)

            if world_size==2:
                loss = F.cross_entropy(out, target)
                loss.backward()
                opt.step()
            elif world_size==1:
                if i%2:
                    loss += F.cross_entropy(out, target)
                else:
                    loss = F.cross_entropy(out, target)
                
                if i%2:
                    (loss/2).backward()
                    opt.step()
        
        net.eval()
        test_out = net(data[train_size:].to(rank))
        pred = test_out.argmax(-1)
        acc = torch.count_nonzero(pred == ds.targets[train_size:].to(rank)) / len(pred)
        test_acc_list.append(acc.cpu().data)

        print(f'rank[{rank}/{world_size-1}] ep[{ep}/{num_epochs}] loss: {loss/2:.4f} testacc: {acc:.4f}')

    
    import numpy as np
    np.save(f'./gpu{world_size}.npy', np.array(test_acc_list))

