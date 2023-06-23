import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os

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
    # load data from matlab file
    mat_data = sio.loadmat(mat_path) 
    # filter dataset
    voxel = mat_data['Volume']
    sample = mat_data['surfaceSamples']
    closest = mat_data['closestPoints']

    return {
      'voxel': torch.from_numpy(voxel).float(),
      'sample': torch.from_numpy(sample).float().t(),
      'closest': torch.from_numpy(closest).float().reshape(-1, 3)
    }
  
  def __len__(self):
    return len(self.data_paths)
  

class ShapeNetLoader():
  def __init__(self, dataset_dir):
    self.dataset = ShapeNetData(dataset_dir)
    self.loader = torch.utils.data.DataLoader(
      self.dataset,
      batch_size=1,
      shuffle=True,
      num_workers=2,
    )
  
  def dataset(self):
    return self.loader

