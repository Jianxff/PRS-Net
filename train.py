import torch
from dataset import ShapeNetLoader
from network import PRSNet
from loss_fn import LossFn
import time


prs_net = PRSNet()
loss_fn = LossFn(weight=25)
data_loader = ShapeNetLoader('/root/autodl-tmp/ShapeNetCore.v2-PT')
dataset = data_loader.dataset()

batch_size = 32
n_iter = 1

optimizer = torch.optim.Adam(prs_net.parameters(), lr=0.001)

for epoch in range(n_iter):
  epoch_tm = time.time()
  for i, data in enumerate(dataset):
    iter_tm = time.time()

    planes, axes = prs_net(data.voxel_grid.voxles)
    loss = loss_fn(data,planes,axes)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'epoch: {epoch}, iter: {i}, loss: {loss}, time: {time.time() - iter_tm}')

  prs_net.save(f'epoch_{epoch}')
