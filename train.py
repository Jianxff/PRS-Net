import torch
from src.dataset import ShapeNetLoader, ShapeNetData
from src.network import PRSNet
from src.loss_fn import LossFn
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r""" Train PRS-Net with ShapeNetCore.v2.train dataset
模型训练
"""

def log(str):
  print(str)
  with open('log/train.log', 'a') as f:
    f.write(str + '\n')

log(f'========== {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ==========')
train_tm = time.time()

batch_size = 32 # batch size
n_epoch = 100 # epoch number

prs_net = PRSNet().to(device)
loss_fn = LossFn(weight = 50).to(device)
data_loader = ShapeNetLoader('/root/autodl-tmp/ShapeNetCore.v2.train',batch_size=batch_size,shuffle=True)

# Adam optimizer with learning rate 0.01
optimizer = torch.optim.Adam(prs_net.parameters(), lr=0.01)

for epoch in range(n_epoch + 1):
  epoch_tm = time.time()
  prs_net = prs_net.to(device)
  loss = None

  for i, data in enumerate(data_loader.dataset()):
    iter_tm = time.time()
    data = ShapeNetData.auto_grad(data)
    planes, axes = prs_net(data['voxel_grid'])
    
    loss = loss_fn(data,planes,axes) # loss
    optimizer.zero_grad() # clear grad
    loss.all.backward() # backward
    optimizer.step() # update parameters

    print(f'[epoch {epoch}, iter {i}] {str(loss)}, time: {(time.time() - iter_tm):.3f}')
  
  if epoch % 10 == 0:
    prs_net.save_network(f'epoch_{epoch}')
  
  log(f'[epoch {epoch}] {str(loss)}, time: {(time.time() - epoch_tm):.3f}')

prs_net.save_network('latest')
log(f'[training] finish training in {(time.time() - train_tm):.3f} seconds')