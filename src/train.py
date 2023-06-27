import torch
from dataset import ShapeNetLoader, ShapeNetData
from network import PRSNet
from loss_fn import LossFn
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r""" Train PRS-Net with ShapeNetCore.v2.train dataset
模型训练
"""

def log(str):
  with open('../train.log', 'a') as f:
    f.write(str + '\n')

log(f'========== {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ==========')

batch_size = 32 # batch size
n_epoch = 100 # epoch number

prs_net = PRSNet().to(device)
loss_fn = LossFn(weight = 25).to(device)
data_loader = ShapeNetLoader('/root/autodl-tmp/ShapeNetCore.v2.train',batch_size)
dataset = data_loader.dataset()

# Adam optimizer with learning rate 0.01
optimizer = torch.optim.Adam(prs_net.parameters(), lr=0.01)

for epoch in range(n_epoch + 1):
  epoch_tm = time.time()
  prs_net = prs_net.to(device)
  loss_str = ''

  for i, data in enumerate(dataset):
    iter_tm = time.time()
    data = ShapeNetData.auto_grad(data)
    planes, axes = prs_net(data['voxel_grid'])
    
    ref_loss, rot_loss, reg_loss = loss_fn(data,planes,axes)
    loss = ref_loss + rot_loss + reg_loss

    optimizer.zero_grad() # clear grad
    loss.backward()       # backward
    optimizer.step()      # update parameters

    loss_str = f'loss: {loss.item():.3f} <ref: {ref_loss.item():.3f}, rot: {rot_loss.item():.3f}, reg: {reg_loss.item():.3f}>'
    print(f'[epoch {epoch}, iter {i}] {loss_str}, time: {(time.time() - iter_tm):.3f}')
  
  if epoch % 10 == 0:
    prs_net.save_network(f'epoch_{epoch}')
  
  log(f'[epoch {epoch}] {loss_str}, time: {(time.time() - epoch_tm):.3f}')

prs_net.save_network('latest')