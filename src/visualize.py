import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
import random


class ShapeNetData:
  data: dict
  ax: plt.Axes

  def load_mat(self, path):
    self.data = sio.loadmat(path)

  def rand_point(self):
    reg = 1 / 32
    px = -0.5 + reg * random.randint(0, 31)
    py = -0.5 + reg * random.randint(0, 31)
    pz = -0.5 + reg * random.randint(0, 31)
    return np.array([px, py, pz])
  
  def init(self):
    self.ax = plt.axes(projection='3d')
    # set axis range:
    self.ax.set_xlim3d(-0.5, 0.5)
    self.ax.set_ylim3d(-0.5, 0.5)
    self.ax.set_zlim3d(-0.5, 0.5)

  def visual_cloud_point(self):
    vtx = np.concatenate((self.data['vertices'], self.data['sample_points']), axis=0)
    x, y, z = vtx[:, 0], vtx[:, 1], vtx[:, 2]
    self.ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

  def visual_closest_point(self, points:np.ndarray = np.array([]), rand_num:int = 1):
    # random point
    if points.shape[0] == 0:
      points = np.array([self.rand_point() for i in range(rand_num)])
    if points.shape[0] == 0:
      return
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    self.ax.scatter(px, py, pz, c='r', marker='^')

    point_d = np.floor((points + 0.5) * 32).astype(int)
    d = self.data['closest_points'][point_d[:, 0], point_d[:, 1], point_d[:, 2]]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    self.ax.scatter(dx, dy, dz, c='g', marker='^')
    # draw line
    for i in range(points.shape[0]):
      self.ax.plot([px[i], dx[i]], [py[i], dy[i]], [pz[i], dz[i]], c='r')


  def visual_symmetry(self):
    # symmetry plane
    for plane in self.data['planes']:
      x, y = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
      z = (-plane[0] * x - plane[1] * y - plane[3]) * 1. / plane[2]
      self.ax.plot_surface(x, y, z, alpha=0.2)
  
  def visual_rotate(self):
    for axis in self.data['axes']:
      # self.ax.plot([0, axis[0]], [0, axis[1]], [0, axis[2]], c='r')
      # TODO
      pass

  def show(self):
    plt.show()

  def visual_all(self):
    self.init()
    self.visual_cloud_point()
    self.visual_closest_point(rand_num=0)
    self.visual_symmetry()
    self.show()


def visualize(path):
  data = ShapeNetData()
  for p in os.listdir(path):
    if p.endswith('.mat'):
      data.load_mat(os.path.join(path, p))
      data.visual_all()

visualize(r'C:\Users\jianxff\OneDrive\桌面\ShapeNetCore.v2-RS')

  