import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
import random

r""" Visualization for PRS-Net
数据可视化
"""

def quat_mul(q1,q2):
  # q = np.zeros((q1.shape[0],4))
  qr = q1[:,0]*q2[:,0] - np.sum(q1[:,1:]*q2[:,1:],axis=1)
  qi = q1[:,0].reshape(-1,1)*q2[:,1:] + q2[:,0].reshape(-1,1)*q1[:,1:] + np.cross(q1[:,1:],q2[:,1:])
  return np.concatenate((qr.reshape(-1,1),qi.reshape(-1,3)),axis=1)


class ShapeNetData:
  r""" ShapeNet dataset
  """

  vetices: np.ndarray
  sample_points: np.ndarray
  closest_points: np.ndarray
  planes: np.ndarray
  quats: np.ndarray

  def load_mat(self, path):
    data = sio.loadmat(path)
    self.vertices = data['vertices']
    self.sample_points = data['sample_points']
    self.vtx = np.concatenate((self.vertices, self.sample_points), axis=0)
    self.closest_points = data['closest_points']
    self.planes = data['planes'][:,0,:]
    self.quats = data['axes'][:,0,:]

  def closest(self,points):
    r""" Get closest points from input points
    根据输入点获取最近点
    """

    idx = np.floor((points + 0.5) * 32).astype(int)
    return self.closest_points[idx[:, 0], idx[:, 1], idx[:, 2]]
  
  def reflect(self, points, plane_id):
    r""" Get reflected points from input points
    根据输入点获取对称点
    """

    plane = self.planes[plane_id]
    dst = 2 * ((np.matmul(points,plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])**2 ).reshape(-1,1)
    points_t = points - dst * plane[:3].reshape(-1,3)
    return points_t
  
  def rotate(self, points, quat_id):
    r""" Get rotated points from input points
    根据输入点获取旋转点
    """

    quat = self.quats[quat_id]
    quat /= np.linalg.norm(quat)
    quat = quat[np.newaxis,:]
    quat_inv = quat * np.array([1,-1,-1,-1])
    points = np.concatenate((np.zeros((points.shape[0],1)),points),axis=1)
    
    points_t = quat_mul(quat,points)
    points_t = quat_mul(points_t,quat_inv)

    points_t = points_t[:,1:]
    return points_t

  
  def axis(self, quat_id):
    r""" Get axis from quaternion
    四元数转旋转轴
    """
    
    quat = self.quats[quat_id]
    quat /= np.linalg.norm(quat)
    theta = 2 * np.arccos(quat[0])
    axis = quat[1:] / np.sin(theta / 2)
    return axis


def init_plt(sub = False):
  r""" Initialize matplotlib
  初始化matplotlib图像
  """
 
  def set_lim(ax):
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.set_aspect('equal')
  if sub:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    for ax in axes: set_lim(ax)
    return axes
  else:
    plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    set_lim(ax)
    return ax

def visual_scatter(ax, points,color='b',cmap=None, linewidth=0.5, marker='o', s=10):
  r""" Visualize scatter points
  可视化散点
  """
  
  if cmap is not None:
    color = points[:, 2]
  ax.scatter(points[:, 0], points[:, 1], points[:, 2],s=s,c=color,cmap=cmap, linewidth=linewidth, marker=marker)

def visual_plane(ax, plane):
  r""" Visualize plane
  可视化平面
  """
  
  x, y = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
  z = (-plane[0] * x - plane[1] * y - plane[3]) * 1. / plane[2]
  ax.plot_surface(x, y, z, alpha=0.2)

def visual_axis(ax, axis,color='r'):
  r""" Visualize axis
  可视化旋转轴
  """

  zr = np.zeros(3)
  end_point = axis
  start_point = zr - end_point
  ax.plot3D([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color)

def visual_line(ax,p1,p2,color='r'):
  r""" Visualize line
  可视化线段
  """
  for i in range(p1.shape[0]):
    ax.plot3D([p1[i,0], p2[i,0]], [p1[i,1], p2[i,1]], [p1[i,2], p2[i,2]], color=color)


def filter(path,limit=-1,rand=False):
  r""" Filter mat files
  过滤mat文件
  """
  files = []
  cnt = 0
  pts = os.listdir(path)
  # if rand: random.shuffle(pts)
  for p in pts:
    if limit >= 0 and cnt >= limit: break
    if p.endswith('.mat'):
      files.append(os.path.join(path, p))
      cnt += 1
  if rand: random.shuffle(files)
  return files


def visualize(path,rand=False,limit=-1,show=True,save=False,save_path='./save'):
  r""" Visualize mat files
  可视化mat文件
  """
  mat = filter(path,limit=limit,rand=rand)
  data = ShapeNetData()
  for m in mat:
    data.load_mat(m)
    ax = init_plt()
    visual_scatter(ax,data.vtx,cmap='viridis') # visualize original model
    # process ==========================================
    
    # visualize symmetry and closest point =============
    visual_plane(ax,data.planes[1])
    p = np.array([data.sample_points[500]])
    ps = data.reflect(p,1)  # reflect
    # ps = data.rotate(ps,1)  # rotate
    pc = data.closest(ps)
    visual_scatter(ax,p,color='r',s=40,marker='v')  # original point
    visual_scatter(ax,ps,color='b',s=40,marker='v') # symmetry point
    visual_scatter(ax,pc,color='c',s=40,marker='v') # closest point
    visual_line(ax,p,ps,color='r')  # point to symmetry point
    visual_line(ax,ps,pc,color='k') # symmetry point to closest point

    # visualize planes and axes  =======================
    for id in range(3):
      visual_plane(ax,data.planes[id])
      visual_axis(ax,data.axis(id))
    
    # show and save ====================================
    if show: plt.show()
    if save: plt.savefig(os.path.join(save_path,os.path.basename(m).split('.')[0]+'.png'))
    plt.close()


visualize(path=r'C:\Users\jianxff\OneDrive\桌面\ShapeNetCore.v2-RS',
          save_path=r'C:\Users\jianxff\OneDrive\桌面\ShapeNetCore.v2-VL',
          rand=True,
          limit=10)

