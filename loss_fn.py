import torch
import numpy as np


# transform 
def transform_reflect(sample,plane):
  # reflect
  a,b,c,d = plane
  sample_t = sample - 2 * torch.Tensor([a,b,c]) * (a * sample[:,0] + b * sample[:,1] + c * sample[:,2] + d) / (a**2 + b**2 + c**2)
  return sample_t

def transform_rotate(sample, quat):
  # complete 0 before sample
  sample_t = torch.cat((torch.zeros(sample.shape[0],1), sample), 1)
  # rotate
  quat_inv = torch.cat((quat[:,0].unsqueeze(1), -quat[:,1:]), 1)
  sample_t = quat * sample_t * quat_inv
  return sample_t[:,1:]


# grid index
def point_grid_indexes(points, grid_bound = 0.5, grid_size = 32):
  min = -grid_bound + grid_bound / grid_size
  max = grid_bound - grid_bound / grid_size
  indexes = (points - min) * grid_size / (2 * grid_bound)
  return indexes

# loss function
def symmetry_loss(sample, closest, grid_size = 32):
  idx_3d = point_grid_indexes(closest)
  idx_vec = idx_3d[:,0] * grid_size**2 + idx_3d[:,1] * grid_size + idx_3d[:,2]
  closest_points = torch.gather(closest,1,idx_vec)


def regular_loss(planes,axes):
  # calc planes are list of (a,b,c,d) and M1 is normalize of each plane
  norm = []
  for p in planes:
    n = p[0:3]
    norm.append(n / torch.norm(n))
  M1 = torch.stack(norm, dim=0)
  M1_ = torch.transpose(M1, 1, 2)

  # calc axes are list of (a,b,c,d) and M2 is normalize of each axis
  u = []
  for a in axes:
    u.append(a[1:4])
  M2 = torch.stack(u, dim=0)
  M2_ = torch.transpose(M2, 1, 2)

  A = torch.matmul(M1, M1_) - torch.eye(3)
  B = torch.matmul(M2, M2_) - torch.eye(3)

  # calc Frobenius
  loss = torch.norm(A) + torch.norm(B)
  return loss


def loss_fn(dataset,planes,axes,weight):
  sym_loss = 0
  sample = dataset['sample']
  voxel = dataset['voxel']
  closest = dataset['closest']

  for p in planes:
    points = transform_reflect(sample,p)
    sym_loss += symmetry_loss(points,closest)
  for a in axes:
    points = transform_rotate(sample,a)
    sym_loss += symmetry_loss(points,closest)

  reg_loss = regular_loss(planes,axes)
  return sym_loss + weight * reg_loss
