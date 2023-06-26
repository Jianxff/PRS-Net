import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import scipy.io as sio
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShapeNetData(Dataset):
  def __init__(self, index_path, test=False):
    self.dataset = []
    self.test = test
    with open(index_path, 'r') as f:
      for line in f:
        info = line.split()
        id, obj, mat = info[0], info[1], info[2]
        if os.path.exists(mat):
          self.dataset.append((id, mat))

  def __getitem__(self, index):
    id, mat = self.dataset[index]
    data = sio.loadmat(mat)
    res = {
      'id': id,
      'bound': data['bound'][0][0],
      'grid_size': data['grid_size'][0][0],
      'sample_points': torch.Tensor(data['sample_points']).to(device),
      'voxel_grid': torch.Tensor(data['voxel_grid']).unsqueeze(0).to(device),
      'closest_points': torch.Tensor(data['closest_points']).to(device),
    }
    if self.test:
      res['vertices'] = torch.Tensor(data['vertices']).to(device)
    return res
  
  def __len__(self):
    return len(self.dataset)
  
  @staticmethod
  def auto_grad(data: dict):
    data['voxel_grid'] = Variable(data['voxel_grid'], requires_grad=True)
    data['sample_points'] = Variable(data['sample_points'])
    data['closest_points'] = Variable(data['closest_points'])
    return data

class ShapeNetLoader:
  def __init__(self, index_file, batch_size, shuffle=False, test=False):
    dataset = ShapeNetData(index_file,test)
    self.loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle
    )
  
  def dataset(self):
    return self.loader

