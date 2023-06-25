import torch
from dataset import ShapeNetLoader
from network import PRSNet
from loss_fn import LossFn
from polygon import Polygon


prs_net = PRSNet()
loss_fn = LossFn(weight=25)
dataset = ShapeNetLoader('dataset').data()

batch_size = 32
n_iter = 0

optimizer = torch.optim.Adam(prs_net.parameters(), lr=0.001)

for epoch in range(n_iter):
  for i, data in enumerate(dataset, 0):
    planes, axes = prs_net(data.voxel_grid.voxles)
    loss = loss_fn(data,planes,axes)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
