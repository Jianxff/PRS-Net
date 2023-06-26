import torch
from torch.utils.data import Dataset
import scipy.io as sio
from torch.autograd import Variable
import os
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
      'sample_points': Variable(torch.Tensor(data['sample_points']), requires_grad=True),
      'voxel_grid': Variable(torch.Tensor(data['voxel_grid']).unsqueeze(0), requires_grad=True),
      'closest_points': Variable(torch.Tensor(data['closest_points']), requires_grad=True)
    }
  
  def __len__(self):
    return len(self.data_paths)

class ShapeNetLoader:
  def __init__(self, dataset_dir, batch_size = 32):
    dataset = ShapeNetData(dataset_dir)
    self.loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True
    )
  
  def dataset(self):
    return self.loader

