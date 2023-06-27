import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import scipy.io as sio
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShapeNetData(Dataset):
  r""" Dataset for pretreated data on ShapeNetCore.v2

  ShapeNetCore.v2预处理数据集
  .. note:
    从.mat载入, 包含:
    - id: 模型id
    - bound: 模型边界
    - grid_size: 体素网格大小
    - sample_points: 采样点
    - voxel_grid: 体素网格
    - closest_points: 提速网格对应的最近模型点
    - * vertices: 模型顶点 
    
    *仅在测试时有效
  """

  def __init__(self, index_path, test=False):
    self.dataset = []
    self.test = test
    with open(index_path, 'r') as f:
      for line in f:
        info = line.split()
        id, obj, mat = info[0], info[1], info[2]
        if os.path.exists(mat): self.dataset.append((id, mat))

  def __getitem__(self, index):
    id, mat = self.dataset[index]
    data = sio.loadmat(mat)
    res = {
      'id': id,
      'bound': data['bound'][0][0],
      'vertices': torch.Tensor(data['vertices']).to(device) if self.test else [],
      'grid_size': data['grid_size'][0][0],
      'sample_points': torch.Tensor(data['sample_points']).to(device),
      'voxel_grid': torch.Tensor(data['voxel_grid']).unsqueeze(0).to(device),
      'closest_points': torch.Tensor(data['closest_points']).to(device),
    }
    return res
  
  def __len__(self):
    return len(self.dataset)
  
  @staticmethod
  def auto_grad(data: dict):
    r""" Convert data to autograd form 

    将数据转换为可自动求导的形式
    """
    data['voxel_grid'] = Variable(data['voxel_grid'], requires_grad=True)
    data['sample_points'] = Variable(data['sample_points'])
    data['closest_points'] = Variable(data['closest_points'])
    return data


class ShapeNetLoader:
  r""" DataLoader for pretreated data on ShapeNetCore.v2

  ShapeNetCore.v2数据加载器
  .. note:
    - index_file: 索引文件路径
    - batch_size: batch大小
    - shuffle: 随机乱序
    - test: 测试
  """

  def __init__(self, index_file, batch_size, shuffle=False, test=False):
    dataset = ShapeNetData(index_file,test)
    self.loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
    )
  
  def dataset(self):
    return self.loader

