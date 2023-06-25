import torch
from torch.utils.data import Dataset
import scipy.io as sio
from polygon import Polygon
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
    return Polygon.load(mat_path)
  
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

