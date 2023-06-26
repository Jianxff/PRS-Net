import torch
from dataset import ShapeNetLoader, ShapeNetData
from network import PRSNet
from loss_fn import LossFn
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')

batch_size = 32
n_iter = 30

prs_net = PRSNet()
loss_fn = LossFn(weight=25)
data_loader = ShapeNetLoader('/root/autodl-tmp/ShapeNetCore.v2.train',batch_size)
dataset = data_loader.dataset()

optimizer = torch.optim.Adam(prs_net.parameters(), lr=0.001)

for epoch in range(n_iter):
  epoch_tm = time.time()
  for i, data in enumerate(dataset):
    iter_tm = time.time()
    data = ShapeNetData.auto_grad(data)
    planes, axes = prs_net(data['voxel_grid'])
    loss = loss_fn(data,planes,axes)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'epoch: {epoch}, iter: {i}, loss: {loss}, time: {time.time() - iter_tm}')
  if epoch % 10 == 0:
    prs_net.save_network(f'epoch_{epoch}')
prs_net.save_network('latest')