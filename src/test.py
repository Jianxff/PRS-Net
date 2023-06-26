import torch
from dataset import ShapeNetLoader
from network import PRSNet
from loss_fn import LossFn
import scipy.io as sio
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')

batch_size = 1
limit = 10
save_dir = '/root/autodl-tmp/ShapeNetCore.v2-RS'

prs_net = PRSNet()
loss_fn = LossFn(weight=25)
data_loader = ShapeNetLoader('/root/autodl-tmp/ShapeNetCore.v2.test', batch_size, shuffle=False)
dataset = data_loader.dataset()

prs_net.load_network('latest')

for i, data in enumerate(dataset):
  if i >= limit:
    break
  iter_tm = time.time()
  planes, axes = prs_net(data['voxel_grid'])
  loss = loss_fn(data,planes,axes)

  result = {
    'id': data['id'],
    'vertices': data['vertices'].cpu().numpy()[0],
    'sample_points': data['sample_points'].detach().cpu().numpy()[0],
    'closest_points': data['closest_points'].detach().cpu().numpy()[0],
    'planes': planes.detach().cpu().numpy()[:,0,:],
    'axes': axes.detach().cpu().numpy()[:,0,:]
  }

  sio.savemat(f'{save_dir}/{data["id"]}.mat', result)
  print(f'data_id: {data["id"]}, loss: {loss}, time: {time.time() - iter_tm}')

