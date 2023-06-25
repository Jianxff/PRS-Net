import numpy as np
import open3d as o3d
import os
import scipy.io as sio


class Surface:
  data: np.ndarray
  def __init__(self, data: np.ndarray):
    self.data = data

  def dist(self):
    dist_01 = np.linalg.norm(self.data[:,0] - self.data[:,1],axis=1)
    dist_12 = np.linalg.norm(self.data[:,1] - self.data[:,2],axis=1)
    dist_20 = np.linalg.norm(self.data[:,2] - self.data[:,0],axis=1)
    dist = np.stack((dist_01, dist_12, dist_20), axis=1)
    return dist

  def filter(self, max):
    dist = self.dist()

    # filter by point 0 and point 1
    mask_01 = np.array((dist[:,0]) > max)
    data_01, remain = self.data[mask_01], self.data[np.logical_not(mask_01)]

    # filter by point 1 and point 2
    mask_12 = np.array(dist[:,1] > max) & np.logical_not(mask_01)
    data_12, remain = remain[mask_12], remain[np.logical_not(mask_12)]

    # filter by point 2 and point 0
    mask_20 = np.array(dist[:,2] > max) & np.logical_not(mask_01 | mask_12)
    data_20, remain = remain[mask_20], remain[np.logical_not(mask_20)]

    return Surface(remain), Surface(data_01), Surface(data_12), Surface(data_20)
  
  def split(self, type):
    mid = (self.data[:,type] + self.data[:,(type + 1) % 3]) / 2
    return [Surface(np.stack((mid, self.data[:,type], self.data[:,(type + 2) % 3]), axis=1)), 
            Surface(np.stack((mid, self.data[:,(type + 1) % 3], self.data[:,(type + 2) % 3]), axis=1))]
    

class VoxelGrid:
  size: int
  voxels: np.ndarray
  def __init__(self, grid_size):
    self.size = grid_size
    self.voxels = np.zeros((grid_size + 1, grid_size + 1, grid_size + 1), dtype=bool)

  def index3d(self, points: np.ndarray, bias = 0):
    idx = np.floor(points  + bias).astype(int)
    idx = np.where((idx >= 0) & (idx < self.size), idx, self.size)
    return idx
  
  def fill_iter(self, surface: Surface, limit = 1):
    remain, s01, s12, s20 = surface.filter(limit)
    self.padding(remain)
    for type, s in [(0,s01), (1,s12), (2,s20)]:
      new_s_1, new_s_2 = s.split(type)
      self.fill_iter(s01, new_s_1)
      self.fill_iter(s12, new_s_2)f 

  def fill(self, surface: Surface, limit = 1):
    self.voxels = self.voxels.pad((0,1),(0,1),(0,1))
    self.fill_iter(surface, limit)
    self.voxels = self.voxels[:-1,:-1,:-1]

  def padding(self, surface: Surface):
    idx = self.index3d(surface.data)
    idx.reshape(-1,3)
    self.voxels[idx] = True


class Polygon:
  # original data
  name: str
  vertices: np.ndarray
  triangles: np.ndarray
  sample_points: np.ndarray
  # voxelize data
  grid_size: int
  bound: float
  voxel_grid: VoxelGrid
  #
  closest_points: np.ndarray

  def __init__(self, grid_size=32, polygon_bound = 0.5):
    self.grid_size = grid_size
    self.bound = polygon_bound
    self.voxel_grid = VoxelGrid(grid_size)
  
  def voxel(self):
    return self.voxel_grid.voxels()

  def load_model(self, path):
    # load data
    mesh = o3d.io.read_triangle_mesh(path)
    self.vertices = np.asarray(mesh.vertices)
    self.triangles = np.asarray(mesh.triangles,dtype=int)
    self.sample_points = np.asarray(mesh.sample_points_uniformly(number_of_points=1000).points)
  
  def scale(self, data):
    return (data + self.bound) * self.grid_size

  def voxelize(self):
    vertices = (self.vertices + self.bound) * self.grid_size
    triangles = self.triangles
    # pre-processing
    vtx0 = vertices[triangles[:,0]]
    vtx1 = vertices[triangles[:,1]]
    vtx2 = vertices[triangles[:,2]]
    surfaces = Surface(np.stack([vtx0, vtx1, vtx2], axis=1))
    self.voxel_grid.fill(surfaces)
    return self.voxel_grid.voxels()

  def compute_closests(self):
    # compute center cube
    seg = 1 / self.grid_size
    col = np.arange(-self.bound + seg / 2 , self.bound - seg /2 + seg, seg)
    x, y, z = np.meshgrid(col, col, col)
    center_cube = self.scale(np.stack([x, y, z], axis=3))
    # init closest points
    cp = np.array([])
    # compute closest points
    center_cube = center_cube.reshape(-1,32,3)
    for k in center_cube:
      diff = self.vertices - k[:, np.newaxis]
      dist = np.linalg.norm(diff, axis=2)
      k_cp_index = np.argmin(dist, axis=1)
      k_cp = self.vertices[k_cp_index]
      np.append(cp, k_cp)
    cp.reshape(self.grid_size, self.grid_size, self.grid_size, 3)
    self.closest_points = cp

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

  @staticmethod
  def load(path):
    data = sio.loadmat(path)
    p = Polygon(data['grid_size'], data['bound'])
    p.voxel_grid.voxels = data['voxel_grid']
    p.sample_points = data['sample_points']
    p.closest_points = data['closest_points']
    # p.vertices = data['vertices']
    # p.triangles = data['triangles']
    return p

