import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import os
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ShapeNetData(Dataset):
  def __init__(self, dataset_dir):
    self.root_dir = dataset_dir
    self.data_paths = []
    # list all items at dir of dataset
    for file in os.listdir(dataset_dir):
      if file.endswith(".mat"):
        self.data_paths.append(os.path.join(dataset_dir, file))

  def __getitem__(self, index):
    mat_path = self.data_paths[index]
    data = sio.loadmat(mat_path)
    return {
      'bound': data['bound'][0][0],
      'grid_size': data['grid_size'][0][0],
      'sample_points': torch.Tensor(data['sample_points']),
      'voxel_grid': torch.Tensor(data['voxel_grid']),
      'closest_points': torch.Tensor(data['closest_points'])
    }
  
  def __len__(self):
    return len(self.data_paths)

class ShapeNetLoader:
  def __init__(self, dataset_dir):
    dataset = ShapeNetData(dataset_dir)
    self.loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=1,
      shuffle=False
    )
  
  def dataset(self):
    return self.loader

