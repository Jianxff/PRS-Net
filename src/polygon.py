import numpy as np
import torch
import scipy.io as sio
from scipy.spatial.transform import Rotation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import trimesh

class Surface:
  r""" Surface data structure for voxel grid

  三角平面
  .. note:
    - dist: 三角距离计算
    - filter/split: 三角平面分割
  """
  data: torch.Tensor
  def __init__(self, data: torch.Tensor = torch.Tensor([])):
    self.data = data

  def __add__(self, other):
    if self.valid():
      return Surface(torch.cat((self.data, other.data), dim=0)) if other.valid() else self
    else:
      return other

  def valid(self):
    return self.data.shape[0] > 0

  def dist(self):
    dist_01 = torch.norm(self.data[:,0] - self.data[:,1],dim=1)
    dist_12 = torch.norm(self.data[:,1] - self.data[:,2],dim=1)
    dist_20 = torch.norm(self.data[:,2] - self.data[:,0],dim=1)
    dist = torch.stack((dist_01, dist_12, dist_20), dim=1)
    return dist

  def filter(self, max):
    dist = self.dist()
    # filter by point 0 and point 1
    mask = [torch.Tensor(dist[:,0] > max).to(device), 
            torch.Tensor(dist[:,1] > max).to(device), 
            torch.Tensor(dist[:,2] > max).to(device)]
    f_mask = [torch.logical_not(mask[0]).to(device), 
              torch.logical_not(mask[1]).to(device), 
              torch.logical_not(mask[2]).to(device)]
    data_mask = [mask[0], (mask[1] & f_mask[0]), (mask[2] & f_mask[1] & f_mask[0])]

    splited = Surface()
    for type in range(0,3):
      s = Surface(self.data[data_mask[type]]).split(type)
      splited = splited + s

    remain = self.data[f_mask[2] & f_mask[1] & f_mask[0]]
    return Surface(remain), splited
  
  def split(self, type):
    if not self.valid(): return Surface()
    mid = (self.data[:,type] + self.data[:,(type + 1) % 3]) / 2
    s1 = torch.stack((mid, self.data[:,type], self.data[:,(type + 2) % 3]), dim=1).to(device)
    s2 = torch.stack((mid, self.data[:,(type + 1) % 3], self.data[:,(type + 2) % 3]), dim=1).to(device)
    return Surface(torch.cat((s1, s2), dim=0))
    

class VoxelGrid:
  r""" Voxel grid data structure

  体素网格
  .. note:
    - index3d: 三维索引 p(x,y,z) -> [(p + Bound) * GridSize]
    - fill: 三角面填充
    超出边界的点将被填充到临时扩展的1个padding上
  """
  size: int
  voxels: np.ndarray
  def __init__(self, grid_size):
    self.size = grid_size
    self.voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

  def index3d(self, points, bias = 0):
    idx = np.floor(points  + bias).astype(int)
    idx = np.where((idx >= 0) & (idx < self.size), idx, self.size)  # fill to temp padding
    return idx
  
  def fill_iter(self, surface: Surface, limit = 1):
    s_ready, s_wait = Surface(), surface
    while(s_wait.valid()):
      # split while trianlge number < 500000
      if s_wait.data.shape[0] < 500000:
        # 3 times split
        for i in range(3):
          if s_wait.valid():
            s1, s2 = s_wait.filter(limit)
            s_ready, s_wait = s_ready + s1, s2
          else:
            break
      else:
        self.padding(s_wait)
        s_wait = Surface()
      self.padding(s_ready)
      s_ready = Surface()

  def fill(self, surface: Surface, limit = 1, iter = True):
    r""" Fill voxel grid with surface

    体素网格填充
    中间处理: 增加1个临时padding边界
    """
    self.voxels = np.pad(self.voxels, ((0,1),(0,1),(0,1)), 'constant', constant_values=False)
    if not iter:
      self.padding(surface)
    else:
      self.fill_iter(surface, limit)
    self.voxels = self.voxels[:-1,:-1,:-1]  # remove padding

  def padding(self, surface: Surface):
    if not surface.valid(): return
    idx = self.index3d(surface.data.cpu().numpy())
    idx.reshape(-1,3)
    self.voxels[idx[:,:,0], idx[:,:,1], idx[:,:,2]] = True


class Polygon:
  r""" Polygon data structure

  多面体
  .. note:
    - id: 多面体id
    - bound: 模型边界
    - grid_size: 体素网格大小
    - vertices: 顶点
    - triangles: 三角面
    - sample_points: 采样点(1000 samples)
    - voxel_grid: 体素网格
    - closest_points: 最近点
  """
  id: str
  # original data
  bound: float
  vertices: np.ndarray
  triangles: np.ndarray
  sample_points: np.ndarray
  # voxelize data
  grid_size: int
  voxel_grid: VoxelGrid
  # closest points
  closest_points: np.ndarray

  def __init__(self, id='polygon', grid_size=32, polygon_bound = 0.5):
    self.id = id
    self.grid_size = grid_size
    self.bound = polygon_bound

  def load_model(self, path, rand_rotate:bool=False):
    r""" Load model from file

    从文件中加载obj模型
    .. note:
      rand_rotate: 随机旋转
    """
    try:
      mesh = trimesh.load(path, force='mesh')
      # mesh = o3d.io.read_triangle_mesh(path)
      self.vertices = np.asarray(mesh.vertices)
      self.triangles = np.asarray(mesh.faces,dtype=int)
      sample, _ = trimesh.sample.sample_surface(mesh, 1000) # 均匀采样
      self.sample_points = np.asarray(sample)
      if self.sample_points.shape[0] != 1000:
        return False
      if rand_rotate:
        R = Rotation.random().as_matrix()
        self.vertices = (np.matmul(R, self.vertices.transpose())).transpose()
        self.sample_points = (np.matmul(R, self.sample_points.transpose())).transpose()
      return True
    except Exception as e:
      print(e)
      return False

  def voxelize(self):
    r""" Voxelization

    体素化
    """
    self.voxel_grid = VoxelGrid(self.grid_size)
    # print('voxelizing...')
    vtx = (self.vertices + self.bound) * self.grid_size
    trg = self.triangles
    # pre-processing
    s_points = np.stack([vtx[trg[:,0]], vtx[trg[:,1]], vtx[trg[:,2]]], axis=1)
    # first group of  triangles
    surface = Surface(torch.Tensor(s_points).to(device))
    # filling voxel grid
    self.voxel_grid.fill(surface)
    return self.voxel_grid.voxels

  def compute_closests(self):
    r""" Compute closest points

    计算最近点
    每 grid_size**2 为一组计算最近点
    .. note:
      - center cube: 等分模型位置 -0.5 ~ 0.5 / grid_size
    """

    # compute center cube
    seg = 1 / self.grid_size
    col = np.arange(-self.bound + seg / 2 , self.bound + seg / 2, seg)
    x, y, z = np.meshgrid(col, col, col)
    center_cube = np.stack([x, y, z], axis=3)
    # init closest points
    self.closest_points = torch.Tensor(self.grid_size,self.grid_size,self.grid_size,3).to(device)
    # compute closest points
    vertices = torch.cat([torch.Tensor(self.vertices), torch.Tensor(self.sample_points)], dim=0).to(device)
    cube = torch.Tensor(center_cube).reshape(-1,self.grid_size**2, 3).to(device)

    for k in cube:
      diff = vertices - k[:, np.newaxis]
      dist = torch.norm(diff, dim=2).to(device)
      k_cp_index = torch.Tensor(torch.argmin(dist, dim=1)).to(device)
      k_cp = vertices[k_cp_index]
      k = torch.floor((k + self.bound) * self.grid_size).int().to(device)
      self.closest_points[k[:,0], k[:,1], k[:,2]] = torch.Tensor(k_cp[:,]).to(device)

    self.closest_points = self.closest_points.cpu().numpy()



  def dump(self, path):
    # save to mat:
    data = {
      'bound': self.bound,
      'grid_size': self.grid_size,
      'sample_points': self.sample_points,
      'vertices': self.vertices,
      'triangles': self.triangles,
      'voxel_grid': self.voxel_grid.voxels,
      'closest_points': self.closest_points
    }
    sio.savemat(path, data)


  def process(self, path, rand_rotate:bool):
    r""" Process model
    集成处理
    模型载入 + 体素化 + 最近点计算
    """
    if not self.load_model(path, rand_rotate):
      return False
    self.voxelize()
    self.compute_closests()
    return True