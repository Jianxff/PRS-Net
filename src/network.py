import torch
import torch.nn as nn
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class PRSNet(nn.Module):
  def __init__(self):
    super(PRSNet, self).__init__()
    # convolution 3d
    self.conv3d = self.conv_layer()
    # linear
    self.linear_reflect = []
    self.linear_rotate = []

    # linear 
    bias_reflect = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    bias_rotate = [[0,0,0,np.sin(np.pi/2)], [0,0,np.sin(np.pi/2),0], [0,np.sin(np.pi/2),0,0]]
    for i in range(3):
      # symmetry reflection
      self.linear_reflect.append(self.linear_layer(bias_reflect[i]))
      # symmetry rotation
      self.linear_rotate.append(self.linear_layer(bias_rotate[i]))


  def forward(self, voxel):
    # convolution 3d 
    v_conved = self.conv3d(voxel)
    v_conved = v_conved.reshape(v_conved.size(0),-1)

    # linear calculate
    planes = torch.stack((self.linear_reflect[0](v_conved),
                          self.linear_reflect[1](v_conved),
                          self.linear_reflect[2](v_conved)),dim=0)
    axes = torch.stack((self.linear_rotate[0](v_conved),
                        self.linear_rotate[1](v_conved),
                        self.linear_rotate[2](v_conved)),dim=0)
    # axes = torch.Tensor([])
    # for i in range(3):
    #   planes = torch.cat((planes, self.linear_reflect[i](v_conved)),dim=1)
    #   axes = torch.cat((axes, self.linear_rotate[i](v_conved)),dim=1)

      # axes.append(self.linear_rotate[i](v_conved))
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
    path = f'../checkpoint/prs_net_{label}.pth'
    torch.save(self.cpu().state_dict(), path)

  def load_network(self, label):
    path = f'../checkpoint/prs_net_{label}.pth'
    self.load_state_dict(torch.load(path))

    