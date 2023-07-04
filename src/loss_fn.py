import torch
from torch import Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnt = 0

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
  * 该部分参考了官方仓库实现
  """

  @staticmethod
  def index(points: Tensor, polygon:dict):
    r""" Get voxel grid index

    获取体素网格索引
    """
    bound, grid_size = polygon['bound'][0], polygon['grid_size'][0]
    indices = torch.floor((points + bound) * grid_size).long()
    indices = torch.clamp(indices, min=0, max=grid_size-1).float()
    w = torch.FloatTensor([grid_size**2, grid_size, 1]).to(device)
    return torch.matmul(indices, w).long()


  @staticmethod
  def closest(indices: Tensor, polygon: dict):
    r""" Get closest points from polygon
    从多边形中获取最近的点
    """
    # get data
    grid_size = polygon['grid_size'][0]
    closest_points = polygon['closest_points'].reshape(-1,grid_size**3,3)
    return torch.gather(closest_points, 1, indices.unsqueeze(-1).repeat(1,1,3))

  @staticmethod
  def mask(indices: Tensor, polygon: dict):
    r""" Get mask points from polygon

    从多边形中获取掩码点
    """
    grid_size = polygon['grid_size'][0]
    voxel_grid = polygon['voxel_grid'].reshape(-1,grid_size**3)
    return torch.gather(voxel_grid, 1, indices)
  

  @staticmethod
  def forward(ctx,points: Tensor, polygon: dict, bias = 0, weight: float = 1):
    r""" Forward propagation
    前向传播, 计算平均距离损失
    """
    idx = DistCount.index(points, polygon)
    closest = DistCount.closest(idx, polygon)
    mask = DistCount.mask(idx, polygon)
    # calc number of 1 in mask
    # num = torch.sum(mask, dim=1).to(device)

    dist = (points - closest)  * (1 - mask).unsqueeze(-1).repeat(1,1,3)
    

    # save for context
    ctx.constant = weight
    ctx.save_for_backward(dist)

    # return loss
    #### the norm_dist is not sqrt() in the official repo, provisionally keep the same here
    dist2 = torch.pow(dist, 2).sum(2).to(device)
    loss = torch.sum(dist2 - bias, dim=1)  

    # with open('temp.txt', 'w') as f:
    #   for i in range(mask.shape[1]):
    #     f.write(f'{int(1 - mask[0][i])} -- {dist2[0][i]} -- [{points[0][i]}] -> [{closest[0][i]}]\n')

    return torch.mean(loss) * weight
  
    
  

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
  
  def reflect_loss(self, plane: Tensor, polygon: dict, bias = 0):
    r""" Accumulate reflect loss

    累计反射平面对称损失
    reflect: p -> p - 2 * (p * n + d) / (||n|| ** 2 + 1e-12)
    """

    samples = polygon['sample_points']
    plane = plane.unsqueeze(1)
    n, d = plane[:,:,:-1], plane[:,:,3] # get normal and bias
    
    dst = 2 * (torch.sum(samples * n, dim=2) + d) / (torch.norm(n, dim=2).pow(2) + 1e-12)
    dst = dst.unsqueeze(2)
    points_t = samples - dst * n
    return DistCount.apply(points_t, polygon, bias)

  def rotate_loss(self, quat: Tensor, polygon: dict, bias = 0):
    r""" Accumulate rotate loss

    累计旋转轴对称损失
    rotate: p -> q * p' * q_inv
    p': (0, p)

    .. note:
      *q_inv: (q_r, -q_i) (*q is normalized)
      避免旋转角度过小, 加上角度的倒数作为损失的一部分
    """

    samples = polygon['sample_points']
    # points -> (0, points)
    points = torch.cat((torch.zeros(samples.shape[0],samples.shape[1],1).to(device), samples), dim=2)
    # rotate
    quat_u = quat.unsqueeze(1)
    quat_inv = quat_u * torch.tensor([1,-1,-1,-1]).to(device)
    # multiply
    points_t = quat_muliply(quat_u, points)
    points_t = quat_muliply(points_t, quat_inv)
    # quat to points
    points_t = points_t[:,:,1:]

    # rotation angle enhance ===============================================
    #### to avoid the angle is too small, adding the reciprocal of the angle
    theta = torch.acos(quat_u[:,:,0]) * 2 * 180 / torch.pi
    theta = torch.where(theta > 180, 360 - theta, theta)
    # ======================================================================

    return DistCount.apply(points_t, polygon, bias) + torch.reciprocal(theta + 1e-12).mean()

  def __call__(self, polygon: dict, planes, axes):
    r""" Call function

    平面反射对称损失 + 旋转轴对称损失
    """
    ref_loss = []
    rot_loss = []

    # 实际上原模型采样点通过该计算也有损失，可以加以考虑
    # samples = polygon['sample_points']
    # bias = (samples - DistCount.closest(samples, polygon)).pow(2).to(device)

    for p in planes:  # iter every plane
      ref_loss.append(self.reflect_loss(p, polygon))
    for a in axes:  # iter every axis (quat)
      rot_loss.append(self.rotate_loss(a, polygon))

    return torch.stack(ref_loss,dim=0), torch.stack(rot_loss,dim=0)


class RegularLoss(torch.nn.Module):
  r""" Regular loss function

  正则化损失函数
  衡量相似度
  """
  def __init__(self):
    super(RegularLoss, self).__init__()

  def list_transpose(self,data:torch.Tensor):
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
    planes, axes = self.list_transpose(planes), self.list_transpose(axes)
    mat_I = torch.eye(3).unsqueeze(0).to(device)
    n_vec = planes[:,:,:-1]
    n_vec = n_vec / (torch.norm(n_vec, dim=2).unsqueeze(2) + 1e-12)
    M1 = n_vec
    M1_T = torch.transpose(M1,1,2)

    u_vec = axes[:,:,1:]
    M2 = u_vec
    M2_T = torch.transpose(M2,1,2)
    A = torch.matmul(M1[:,], M1_T[:,]) - mat_I
    B = torch.matmul(M2[:,], M2_T[:,]) - mat_I
    loss = torch.mean(A.pow(2).sum(2).sum(1) + B.pow(2).sum(2).sum(1))
    return loss

class Loss:
  r""" Loss class
  """
  def __init__(self, ref_loss_list, rot_loss_list, reg_loss):
    self.set(ref_loss_list, rot_loss_list, reg_loss)

  def set(self, ref_loss_list, rot_loss_list, reg_loss):
    self.ref_list = ref_loss_list
    self.rot_list = rot_loss_list
    self.reg = reg_loss
    self.ref = ref_loss_list.sum()
    self.rot = rot_loss_list.sum()
    self.all = self.ref + self.rot + self.reg

  def dump(self):
    return {
      'reflect': self.ref_list.detach().cpu().numpy().tolist(),
      'rotate': self.rot_list.detach().cpu().numpy().tolist(),
      'regular': self.reg.detach().cpu().numpy(),
      'all': self.all.detach().cpu().numpy()
    }

  def __str__(self):
    ref_str = f'ref: {self.ref_list[0].item():.4f} + {self.ref_list[1].item():.4f} + {self.ref_list[2].item():.4f}'
    rot_str = f'rot: {self.rot_list[0].item():.4f} + {self.rot_list[1].item():.4f} + {self.rot_list[2].item():.4f}'
    return f'loss: {self.all.item():.4f} <reg: {self.reg.item():4f}, {ref_str}, {rot_str}>'



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
    return Loss(ref_loss, rot_loss, reg_loss * self.weight)


