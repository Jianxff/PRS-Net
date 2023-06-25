import torch
from torch import Tensor
import numpy as np

class DistCount(torch.autograd.Function):
  @staticmethod
  def forward(ctx, points: Tensor, polygon: dict, weight: float = 1):
    # idx = np.floor(points  + bias).astype(int)
    # idx = np.where((idx >= 0) & (idx < self.size), idx, self.size)
    bound = polygon['bound']
    grid_size = int(polygon['grid_size'])
    closest_points = polygon['closest_points'].reshape(grid_size,grid_size,grid_size,3)
    # voxel_grid = polygon['voxel_grid'].reshape(grid_size,grid_size,grid_size)
    
    indices = torch.floor((points + bound) * grid_size).long()
    indices = torch.clamp(indices, min=0, max=grid_size-1)
    # flag = (0 <= indices[:,0] < grid_size) | (0 <= indices[:,1] < grid_size) | (0 <= indices[:,2] < grid_size)
    # flag = ((indices[:,0] >= 0) & (indices[:,0] < grid_size)) & ((indices[:,1] >= 0) & (indices[:,1] < grid_size)) & ((indices[:,2] >= 0) & (indices[:,2] < grid_size))
    # indices = indices[flag]
    closest = closest_points[indices[:,0], indices[:,1], indices[:,2]]
    # points = points[flag]

    dist = points - closest
    # exceed = points.shape[0] - indices.shape[0]
    # dist = torch.stack()

    # mask = voxel_grid[indices[:,0], indices[:,1], indices[:,2]]
    # print(mask.shape)
    # mask.reshape(-1,1)
    # dist = dist * mask

    ctx.constant = weight
    ctx.save_for_backward(dist)
    norm_dist = torch.norm(dist, dim=1)
    return torch.mean(torch.sum(norm_dist, dim=0)) * weight
  
  @staticmethod
  def backward(ctx, grad_output):
    dist, = ctx.saved_tensors
    grad_trans_points = 2 * (dist) * ctx.constant /(dist.shape[0])
    return grad_trans_points, None, None, None, None

class SymmetryLoss(torch.nn.Module):
  def __init__(self):
    super(SymmetryLoss, self).__init__()
    self.reflect_loss = torch.Tensor([0])
    self.rotate_loss = torch.Tensor([0])
  
  # transform 
  def acc_reflect(self, plane: Tensor, polygon: dict):
    # reflect
    points = polygon['sample_points']
    a,b,c,d = plane
    n = torch.Tensor([a,b,c]).reshape(3,1)
    points = points.reshape(-1,3)
    points_t = points - 2 * (torch.matmul(points,n) + d) / (torch.norm(n)**2) * n.reshape(1,3)
    self.reflect_loss += DistCount.apply(points_t, polygon)

  def acc_rotate(self, quat: Tensor, polygon: dict):
    # complete 0 before sample
    points = polygon['sample_points'].reshape(-1,3)
    points = torch.cat((torch.zeros(points.shape[0],1), points), dim=1)
    # rotate
    quat_inv = quat * torch.tensor([1,-1,-1,-1])
    points_t = quat * points * quat_inv
    points_t = points_t[:,1:]
    self.rotate_loss += DistCount.apply(points_t, polygon)


  def __call__(self, polygon: dict, planes, axes):
    self.reflect_loss = torch.Tensor([0])
    self.rotate_loss = torch.Tensor([0])
    for p in planes:
      self.acc_reflect(p, polygon)
    for a in axes:
      self.acc_rotate(a, polygon)
    return self.reflect_loss + self.rotate_loss


class RegularLoss(torch.nn.Module):
  def __init__(self):
    super(RegularLoss, self).__init__()

  def __call__(self, planes, axes):
    n_vec = planes[:,:-1]
    n_vec = n_vec / torch.norm(n_vec, dim=1)
    M1 = n_vec
    M1_T = torch.transpose(M1,0,1)

    u_vec = axes[:,1:]
    M2 = u_vec
    M2_T = torch.transpose(M2,0,1)

    A = torch.matmul(M1, M1_T) - torch.eye(3)
    B = torch.matmul(M2, M2_T) - torch.eye(3)

    loss = torch.norm(A) + torch.norm(B)
    return loss


class LossFn(torch.nn.Module):
  def __init__(self, weight: float):
    super(LossFn, self).__init__()
    self.weight = weight
    self.sym_loss = SymmetryLoss()
    self.reg_loss = RegularLoss()

  def forward(self, polygon: dict, planes, axes):
    sym_loss = self.sym_loss(polygon,planes, axes)
    reg_loss = self.reg_loss(planes, axes)
    return sym_loss + self.weight * reg_loss

