import torch
from torch import Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DistCount(torch.autograd.Function):
  @staticmethod
  def forward(ctx, points: Tensor, polygon: dict, weight: float = 1):
    bound, grid_size = polygon['bound'][0], polygon['grid_size'][0]
    closest_points = polygon['closest_points'].reshape(-1,grid_size,grid_size,grid_size,3)

    indices = torch.floor((points + bound) * grid_size).long()
    indices = torch.clamp(indices, min=0, max=grid_size-1)
    closest = torch.Tensor([]).to(device)

    for batch in range(points.shape[0]):
      cp, idx  = closest_points[batch,:], indices[batch,:]
      clp = cp[idx[:,0], idx[:,1], idx[:,2]].unsqueeze(0)
      closest = torch.cat((closest, clp), dim=0)
    dist = points - closest
    ctx.constant = weight
    ctx.save_for_backward(dist)
    norm_dist = torch.pow(dist, 2)
    return torch.mean(torch.sum(norm_dist, dim=1)) * weight
  
  @staticmethod
  def backward(ctx, grad_output):
    dist, = ctx.saved_tensors
    grad_trans_points = 2 * (dist) * ctx.constant / (dist.shape[0])
    return grad_trans_points, None, None, None, None

class SymmetryLoss(torch.nn.Module):
  def __init__(self):
    super(SymmetryLoss, self).__init__()
  
  # transform 
  def acc_reflect(self, plane: Tensor, polygon: dict):
    # reflect
    points = polygon['sample_points']
    plane = plane.unsqueeze(1)
    n = plane[:,:,:-1]
    d = plane[:,:,3]
    
    dst = 2 * (torch.sum(points * n, dim=2) + d) / (torch.norm(n, dim=2) ** 2 + 1e-12)
    dst = dst.unsqueeze(2)
    points_t = points - dst * n
    return DistCount.apply(points_t, polygon)

  def acc_rotate(self, quat: Tensor, polygon: dict):
    # complete 0 before sample
    points = polygon['sample_points']
    points = torch.cat((torch.zeros(points.shape[0],points.shape[1],1).to(device), points), dim=2)
    # rotate
    quat_u = quat.unsqueeze(1)
    quat_inv = quat_u * torch.tensor([1,-1,-1,-1]).to(device)
    points_t = quat_u * points * quat_inv
    points_t = points_t[:,:,1:]
    return DistCount.apply(points_t, polygon)


  def __call__(self, polygon: dict, planes, axes):
    ref_loss = torch.Tensor([0]).to(device)
    rot_loss = torch.Tensor([0]).to(device)
    for p in planes:
      loss = self.acc_reflect(p, polygon)
      ref_loss += loss
    for a in axes:
      rot_loss += self.acc_rotate(a, polygon)
    return ref_loss, rot_loss


class RegularLoss(torch.nn.Module):
  def __init__(self):
    super(RegularLoss, self).__init__()

  def acc_transpose(self,data:torch.Tensor):
    out = torch.Tensor([]).to(device)
    for d in data:
      d = d.unsqueeze(1)
      out = torch.cat((out,d), dim=1)
    return out


  def __call__(self, planes, axes):
    planes, axes = self.acc_transpose(planes), self.acc_transpose(axes)
    mat_I = torch.eye(3).unsqueeze(0).to(device)
    n_vec = planes[:,:,:-1]
    n_vec = n_vec / torch.norm(n_vec, dim=2).unsqueeze(2)
    M1 = n_vec
    M1_T = torch.transpose(M1,1,2)

    u_vec = axes[:,:,1:]
    M2 = u_vec
    M2_T = torch.transpose(M2,1,2)
    A = torch.matmul(M1[:,], M1_T[:,]) - mat_I
    B = torch.matmul(M2[:,], M2_T[:,]) - mat_I
    loss = torch.mean(torch.norm(A, dim=1)**2 + torch.norm(B, dim=1)**2)
    return loss


class LossFn(torch.nn.Module):
  def __init__(self, weight: float):
    super(LossFn, self).__init__()
    self.weight = weight
    self.sym_loss = SymmetryLoss()
    self.reg_loss = RegularLoss()

  def forward(self, polygon: dict, planes, axes):
    ref_loss, rot_loss = self.sym_loss(polygon,planes, axes)
    reg_loss = self.reg_loss(planes, axes)
    return ref_loss, rot_loss, reg_loss * self.weight

