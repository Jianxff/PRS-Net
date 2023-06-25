import torch
from torch import Tensor
import numpy as np
from polygon import Polygon


class DistCount(torch.autograd.Function):
  @staticmethod
  def forward(ctx, points: Tensor, polygon: Polygon, weight: float = 1):
    indices = polygon.voxel_grid.index3d(points.numpy())
    closest = polygon.closest_points[indices]
    dist = torch.norm(points - closest, dim=1)

    mask = polygon.voxel_grid.voxels[indices]
    mask.reshape(-1,1)
    
    dist = dist * mask
    ctx.constant = weight
    ctx.save_for_backward(dist)
    return torch.mean(torch.sum(dist, dim=1)) * weight
  
  @staticmethod
  def backward(ctx, grad_output):
    dist, = ctx.saved_tensors
    grad_trans_points = 2 * (dist) * ctx.constant /(dist.shape[0])
    return grad_trans_points, None, None, None, None

class SymmetryLoss(torch.nn.Module):
  def __init__(self):
    self.reflect_loss = torch.Tensor([0])
    self.rotate_loss = torch.Tensor([0])
  
  # transform 
  def acc_reflect(self, plane: Tensor, polygon: Polygon):
    # reflect
    points = polygon.sample_points
    a,b,c,d = plane
    n = torch.Tensor([a,b,c])
    points_t = points - 2 * n * (torch.matmul(points,n) + d) / torch.matmul(n,n)
    self.reflect_loss += DistCount.apply(points_t, polygon)

  def acc_rotate(self, quat: Tensor, polygon: Polygon):
    # complete 0 before sample
    points = polygon.sample_points
    points = torch.Tensor(torch.cat((torch.zeros(points.shape[0],1), points), dim=1))
    # rotate
    quat_inv = quat * torch.tensor([1,-1,-1,-1])
    points_t = quat * points * quat_inv
    points_t = points_t[:,1:]
    self.rotate_loss += DistCount.apply(points_t, polygon)


  def __call__(self, polygon: Polygon, planes, axes):
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
    M1_T = torch.transpose(M1, 1, 2)

    u_vec = axes[:,1:]
    M2 = u_vec
    M2_T = torch.transpose(M2, 1, 2)

    A = torch.matmul(M1, M1_T) - torch.eye(3)
    B = torch.matmul(M2, M2_T) - torch.eye(3)

    loss = torch.norm(A) + torch.norm(B)
    return loss


class LossFn(torch.nn.Module):
  def __init__(self, weight: float):
    self.weight = weight
    self.sym_loss = SymmetryLoss()
    self.reg_loss = RegularLoss()

  def forward(self, polygon: Polygon, planes, axes):
    sym_loss = self.sym_loss(polygon,planes, axes)
    reg_loss = self.reg_loss(planes, axes)
    return sym_loss + self.weight * reg_loss

