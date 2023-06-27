import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PRSNet(nn.Module):
  r""" PRS-Net: Planar Reflective Symmetry Detection Net

  PRS-Net: 平面反射对称检测网络
  .. note:
    - conv3d: 3d convolution
    - linear: fully connected layer
  """
  def __init__(self):
    super(PRSNet, self).__init__()
    # convolution 3d
    self.conv3d = self.conv_layer()
    # linear
    # basic bias 
    bias_reflect = [[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]]
    bias_rotate = [[0,1.,0,0], [0,0,1.,0], [0,0,0,1.]]
    for i in range(3):
      # symmetry reflection
      self.__setattr__(f'linear_plane_{i}', self.linear_layer(bias_reflect[i],clear_weight=True).to(device))
      # symmetry rotation
      self.__setattr__(f'linear_axis_{i}', self.linear_layer(bias_rotate[i]).to(device))


  def forward(self, voxel):
    r""" Forward propagation

    前向传播
    voxel_grid -> plane * 3 & axis * 3
    """
    # convolution 3d 
    v_conved = self.conv3d(voxel)
    v_conved = v_conved.reshape(v_conved.size(0),-1).to(device)

    # linear calculate
    planes = []
    axes = []
    for i in range(3):
      plane = self.__getattr__(f'linear_plane_{i}')(v_conved)
      plane = plane / (1e-12 + torch.norm(plane[:,:3], dim=1).unsqueeze(1))
      planes.append(plane)
      axis = self.__getattr__(f'linear_axis_{i}')(v_conved)
      axis = axis / (1e-12 + torch.norm(axis[:,:], dim=1).unsqueeze(1))
      axes.append(axis)

    return planes, axes


  # convolution 3d layer
  def conv_layer(self):
    in_channels = 1
    out_channels = 4
    model = []
    for i in range(5):
      # convolution
      model += [nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
      # maxpooling and activation
      model += [nn.MaxPool3d(kernel_size=2), nn.LeakyReLU(0.2, inplace=True)]
      # iter channels
      in_channels = out_channels
      out_channels = out_channels * 2
    return nn.Sequential(*model)


  # fully connected layer
  def linear_layer(self, bias, clear_weight = False):
    in_channels = 64
    model = []
    for i in range(2):
      # fully connected layer ans activation
      model += [nn.Linear(in_channels, int(in_channels/2)), nn.LeakyReLU(0.2, inplace=True)]
      in_channels = int(in_channels/2)

    # init bias and weight
    out = nn.Linear(in_channels, 4)
    out.bias.data = torch.Tensor(bias)
    if clear_weight:
      out.weight.data = torch.zeros(4, in_channels)
    
    model += [out]
    return nn.Sequential(*model)
  
  def save_network(self, label):
    path = f'checkpoint/prs_net_{label}.pth'
    torch.save(self.state_dict(), path)

  def load_network(self, label):
    path = f'checkpoint/prs_net_{label}.pth'
    self.load_state_dict(torch.load(path))

    
