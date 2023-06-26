import torch
from torch.utils.data import Dataset
import scipy.io as sio
from torch.autograd import Variable
import os
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ShapeNetData(Dataset):
  def __init__(self, index_path):
    self.dataset = []
    with open(index_path, 'r') as f:
      for line in f:
        info = line.split
        id, obj, mat = info[0], info[1], info[2]
        self.dataset.append((id, mat))

  def __getitem__(self, index):
    id, mat = self.dataset[index]
    data = sio.loadmat(mat)
    return {
      'id': id,
      'bound': data['bound'][0][0],
      'grid_size': data['grid_size'][0][0],
      'vertices': data['vertices'],
      'sample_points': torch.Tensor(data['sample_points']),
      'voxel_grid': torch.Tensor(data['voxel_grid']).unsqueeze(0),
      'closest_points': torch.Tensor(data['closest_points'])
    }
  
  def __len__(self):
    return len(self.data_paths)
  
  @staticmethod
  def auto_grad(data: dict):
    data['voxel_grid'] = Variable(data['voxel_grid'], requires_grad=True)
    data['sample_points'] = Variable(data['sample_points'], requires_grad=True)
    data['closest_points'] = Variable(data['closest_points'], requires_grad=True)
    return data

class ShapeNetLoader:
  def __init__(self, index_file, batch_size, shuffle=False):
    dataset = ShapeNetData(index_file)
    self.loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle
    )
  
  def dataset(self):
    return self.loader

