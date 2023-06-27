import torch
from torch import Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def quat_muliply(q1: Tensor, q2: Tensor):
  r""" Quaternion multiplication

  四元数乘法
  自动展开四元数维度为(-1, 4)
  q1 * q2 = (q_r, q_i)

  q1 = (q1_r, q1_i), q2 = (q2_r, q2_i)
  q_r = q1_r * q2_r - sum(q1_i * q2_i)
  q_i = q1_r * q2_i + q2_r * q1_i + cross(q1_i, q2_i)

  .. note:
    Q1 = (u10, u11, u12, u13), Q2 = (u20, u21, u22, u23)
    Q1 * Q2 = (u10*u20 - u11*u21 - u12*u22 - u13*u23,
               u10*u21 + u11*u20 + u12*u23 - u13*u22,
               u10*u22 - u11*u23 + u12*u20 + u13*u21,
               u10*u23 + u11*u22 - u12*u21 + u13*u20)
  """
  batch = q1.shape[0]
  qr = q1[:,:,0]*q2[:,:,0] - torch.sum(q1[:,:,1:]*q2[:,:,1:],dim=2).to(device)
  qi = q1[:,:,0].reshape(batch,-1,1)*q2[:,:,1:] + q2[:,:,0].reshape(batch,-1,1)*q1[:,:,1:]
  qi += torch.cross(q1[:,:,1:],q2[:,:,1:],dim=2).to(device)
  q = torch.cat((qr.reshape(batch,-1,1), qi.reshape(batch,-1,3)), dim=2).to(device)
  return q.reshape(batch,-1,4)


class DistCount(torch.autograd.Function):
  r""" Distance count loss function

  距离计数损失函数
  体素网格索引: [(Point + Bound) * GridSize] -> (3 * 3) int
  """
  @staticmethod
  def forward(ctx, points: Tensor, polygon: dict, weight: float = 1):
    # get data
    bound, grid_size = polygon['bound'][0], polygon['grid_size'][0]
    closest_points = polygon['closest_points'].reshape(-1,grid_size,grid_size,grid_size,3)
    # get indices
    indices = torch.floor((points + bound) * grid_size).long()
    indices = torch.clamp(indices, min=0, max=grid_size-1)
    closest = torch.Tensor([]).to(device)
    # get closest points from each batch of data
    for batch in range(points.shape[0]):
      cp, idx  = closest_points[batch,:], indices[batch,:]
      clp = cp[idx[:,0], idx[:,1], idx[:,2]].unsqueeze(0)
      closest = torch.cat((closest, clp), dim=0)
    # dist count
    dist = points - closest
    # save for context
    ctx.constant = weight
    ctx.save_for_backward(dist)
    # return loss
    norm_dist = torch.pow(dist, 2).to(device)
    return torch.mean(torch.sum(norm_dist, dim=1)) * weight
  
  @staticmethod
  def backward(ctx, grad_output):
    r""" Backward propagation

    反向传播
    dist = (p - closest_p) ** 2 -> every batch_size
    f' = 2 * (p - closest_p) / batch_size
    """
    dist, = ctx.saved_tensors
    grad_trans_points = 2 * (dist) * ctx.constant / (dist.shape[0])
    return grad_trans_points, None, None, None, None



class SymmetryLoss(torch.nn.Module):
  r""" Symmetry loss function

  对称性损失函数
  包含平面反射对称以及旋转轴对称
  """
  def __init__(self):
    super(SymmetryLoss, self).__init__()
  
  def acc_reflect(self, plane: Tensor, polygon: dict):
    r""" Accumulate reflect loss

    累计反射平面对称损失
    reflect: p -> p - 2 * (p * n + d) / (||n|| ** 2 + 1e-12)
    """

    points = polygon['sample_points']
    plane = plane.unsqueeze(1)
    n, d = plane[:,:,:-1], plane[:,:,3]
    
    dst = 2 * (torch.sum(points * n, dim=2) + d) / (torch.norm(n, dim=2) ** 2 + 1e-12)
    dst = dst.unsqueeze(2)
    points_t = points - dst * n
    return DistCount.apply(points_t, polygon)

  def acc_rotate(self, quat: Tensor, polygon: dict):
    r""" Accumulate rotate loss

    累计旋转轴对称损失
    rotate: p -> q * p' * q_inv
    p': (0, p)

    .. note:
      *q_inv: (q_r, -q_i) (*q is normalized)
    """

    points = polygon['sample_points']
    # points -> (0, points)
    points = torch.cat((torch.zeros(points.shape[0],points.shape[1],1).to(device), points), dim=2)
    # rotate
    quat_u = quat.unsqueeze(1)
    quat_inv = quat_u * torch.tensor([1,-1,-1,-1]).to(device)
    # multiply
    points_t = quat_muliply(quat_u, points)
    points_t = quat_muliply(points_t, quat_inv)
    # quat to points
    points_t = points_t[:,:,1:]
    return DistCount.apply(points_t, polygon)

  def __call__(self, polygon: dict, planes, axes):
    r""" Call function

    平面反射对称损失 + 旋转轴对称损失
    """
    ref_loss = torch.Tensor([0]).to(device)
    rot_loss = torch.Tensor([0]).to(device)
    for p in planes:
      loss = self.acc_reflect(p, polygon)
      ref_loss += loss
    for a in axes:
      rot_loss += self.acc_rotate(a, polygon)
    return ref_loss, rot_loss


class RegularLoss(torch.nn.Module):
  r""" Regular loss function

  正则化损失函数
  衡量相似度
  """
  def __init__(self):
    super(RegularLoss, self).__init__()

  def acc_transpose(self,data:torch.Tensor):
    r""" Accumulate transpose

    batch 转置
    (num, batch_size, 3) -> (batch_size, 3, num)
    """
  
    out = torch.Tensor([]).to(device)
    for d in data:
      d = d.unsqueeze(1)
      out = torch.cat((out,d), dim=1)
    return out


  def __call__(self, planes, axes):
    r""" Call function

    平面正交损失 + 旋转轴正交损失
    plane[n, d] -> M1 = [n1/|n1|, n2/|n2|, n3/|n3|]_T
    axes[u0, u1, u2, u3] -> M2 = [u1/|u1|, u2/|u2|, u3/|u3|]_T
    loss = M * M_T - I
    """
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
  r""" Loss function
  
  对称性损失 + 正则化损失
  """
  def __init__(self, weight: float):
    super(LossFn, self).__init__()
    self.weight = weight
    self.sym_loss = SymmetryLoss()
    self.reg_loss = RegularLoss()

  def forward(self, polygon: dict, planes, axes):
    ref_loss, rot_loss = self.sym_loss(polygon,planes, axes)
    reg_loss = self.reg_loss(planes, axes)
    return ref_loss, rot_loss, reg_loss * self.weight

