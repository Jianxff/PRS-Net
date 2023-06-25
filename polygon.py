import numpy as np
import open3d as o3d
import torch
import scipy.io as sio

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Surface:
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
    mask = [torch.Tensor(dist[:,0] > max), torch.Tensor(dist[:,1] > max), torch.Tensor(dist[:,2] > max)]
    f_mask = [torch.logical_not(mask[0]), torch.logical_not(mask[1]), torch.logical_not(mask[2])]
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
    s1 = torch.stack((mid, self.data[:,type], self.data[:,(type + 2) % 3]), dim=1)
    s2 = torch.stack((mid, self.data[:,(type + 1) % 3], self.data[:,(type + 2) % 3]), dim=1)
    return Surface(torch.cat((s1, s2), dim=0))
    

class VoxelGrid:
  size: int
  voxels: np.ndarray
  def __init__(self, grid_size):
    self.size = grid_size
    self.voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

  def index3d(self, points, bias = 0):
    idx = np.floor(points  + bias).astype(int)
    idx = np.where((idx >= 0) & (idx < self.size), idx, self.size)
    return idx
  
  def fill_iter(self, surface: Surface, limit = 1):
    s_ready, s_wait = Surface(), surface
    while(s_wait.valid()):
      print(f'waiting: {s_wait.data.shape[0]}')
      if s_wait.data.shape[0] < 1000000:
        for i in range(5):
          if s_wait.valid():
            s1, s2 = s_wait.filter(limit)
            s_ready = s_ready + s1
            s_wait = s2
          else:
            break
      else:
        self.padding(s_wait)
        s_wait = Surface()
      self.padding(s_ready)
      s_ready = Surface()


  def fill(self, surface: Surface, limit = 1, iter = True):
    self.voxels = np.pad(self.voxels, ((0,1),(0,1),(0,1)), 'constant', constant_values=False)
    if not iter:
      self.padding(surface)
    else:
      self.fill_iter(surface, limit)
    self.voxels = self.voxels[:-1,:-1,:-1]

  def padding(self, surface: Surface):
    if not surface.valid(): return
    idx = self.index3d(surface.data.cpu().numpy())
    idx.reshape(-1,3)
    self.voxels[idx[:,:,0], idx[:,:,1], idx[:,:,2]] = True


class Polygon:
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

  def __init__(self, grid_size=32, polygon_bound = 0.5):
    self.grid_size = grid_size
    self.bound = polygon_bound
  
  def voxel(self):
    return self.voxel_grid.voxels

  def load_model(self, path):
    # load data
    try:
      mesh = o3d.io.read_triangle_mesh(path)
      self.vertices = np.asarray(mesh.vertices)
      self.triangles = np.asarray(mesh.triangles,dtype=int)
      self.sample_points = np.asarray(mesh.sample_points_uniformly(number_of_points=1000).points)
      return True
    except:
      print("load error")
      return False
  
  def scale(self, data):
    return (data + self.bound) * self.grid_size

  def voxelize(self):
    self.voxel_grid = VoxelGrid(self.grid_size)
    print('voxelizing...')
    vtx = (self.vertices + self.bound) * self.grid_size
    trg = self.triangles
    # pre-processing
    print(f'vertices: {len(vtx)}, triangles: {len(trg)}')
    s_points = np.stack([vtx[trg[:,0]], vtx[trg[:,1]], vtx[trg[:,2]]], axis=1)
    self.voxel_grid.fill(Surface(torch.Tensor(s_points)))
    print('voxelization finished')
    return self.voxel_grid.voxels

  def compute_closests(self):
    print('computing closest points')
    # compute center cube
    seg = 1 / self.grid_size
    col = np.arange(-self.bound + seg / 2 , self.bound - seg /2 + seg, seg)
    x, y, z = np.meshgrid(col, col, col)
    center_cube = self.scale(np.stack([x, y, z], axis=3))
    # init closest points
    cp = torch.Tensor([])
    # compute closest points
    vertices = torch.Tensor(self.vertices)
    cube = torch.Tensor(center_cube)
    cube = cube.reshape(-1,1024,3)
    cnt = 0
    for k in cube:
      cnt += 1
      # print(f'waiting: {32 - cnt}')
      diff = vertices - k[:, np.newaxis]
      dist = torch.norm(diff, dim=2)
      k_cp_index = torch.argmin(dist, dim=1)
      k_cp = vertices[k_cp_index]
      cp = torch.cat([cp, k_cp],dim=0)

    cp = cp.cpu().numpy()
    self.closest_points = cp.reshape(self.grid_size, self.grid_size, self.grid_size, 3)
    print('computation finished')

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
    np.save(path + '.npy', self.voxel_grid.voxels)

