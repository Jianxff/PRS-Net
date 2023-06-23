import torch
from dataset import ShapeNetLoader
from model import PRSNet
from loss_fn import loss_fn


prs_net = PRSNet()

dataset = ShapeNetLoader('dataset').data()

batch_size = 32
n_iter = 0

optimizer = torch.optim.Adam(prs_net.parameters(), lr=0.001)

for epoch in range(n_iter):
  for i, data in enumerate(dataset, 0):
    planes, axes = prs_net(data['voxel'])
    loss = loss_fn(data,planes,axes,weight=1)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
