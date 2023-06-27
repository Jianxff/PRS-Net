import torch
from src.dataset import ShapeNetLoader
from src.network import PRSNet
from src.loss_fn import LossFn
import scipy.io as sio
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

r""" Test PRS-Net with ShapeNetCore.v2.train dataset
模型测试
"""

def log(str):
  with open('log/test.log', 'a') as f:
    f.write(str + '\n')

log(f'========== {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ==========')

limit = 50 # test model limit (*-1 for no limit)
save_dir = '/root/autodl-tmp/ShapeNetCore.v2-RS' # output path

prs_net = PRSNet()
loss_fn = LossFn(weight=25).to(device)
data_loader = ShapeNetLoader('/root/autodl-tmp/ShapeNetCore.v2.test', batch_size=1, shuffle=True, test=True)
dataset = data_loader.dataset()

prs_net.load_network('epoch_0')
prs_net = prs_net.to(device)

for i, data in enumerate(dataset):
  if limit >= 0 and i >= limit:
    break
  iter_tm = time.time()
  planes, axes = prs_net(data['voxel_grid'])
  loss = loss_fn(data,planes,axes)

  result = {
    'id': data['id'][0],
    'loss': loss.dump(),
    'vertices': data['vertices'].cpu().numpy()[0],
    'sample_points': data['sample_points'].detach().cpu().numpy()[0],
    'closest_points': data['closest_points'].detach().cpu().numpy()[0],
    'planes': [plane.detach().cpu().numpy() for plane in planes],
    'axes': [axis.detach().cpu().numpy() for axis in axes],
  }

  sio.savemat(f'{save_dir}/{data["id"][0]}.mat', result)

  print(f'[id {data["id"][0]}] {str(loss)}, time: {(time.time() - iter_tm):.3f}')
  log(f'[id {data["id"][0]}] {str(loss)}, time: {(time.time() - iter_tm):.3f}')

