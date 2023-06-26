
# # import torch
# # # a = torch.tensor([[1,2,3],[2,3,4],[4,5,6]])
# # # z = torch.zeros(a.shape[0],1)
# # # print('a',a)
# # # print('z',z)
# # # inputs = [a,z]
# # # # b = torch.cat(inputs,dim=1)

# # # b = torch.cat((torch.zeros(a.shape[0],1), a), dim=1)
# # # print(b)

# # # a = torch.Tensor([0,0,0])
# # # b = torch.Tensor([]).reshape(-1,3)
# # # c = torch.stack((a,b),dim=0)
# # # print(c)
# # a = torch.Tensor([1,2,3])
# # b = torch.Tensor([4,5,6])
# # c = torch.stack((a,b),dim=0)
# # print(c)

# import open3d as o3d
# import trimesh
# import numpy as np

# path = r'/root/autodl-tmp/ShapeNetCore.v2/03790512/ba0f7f4b5756e8795ae200efe59d16d0/models/model_normalized.obj'
# path2 = r'/root/autodl-tmp/ShapeNetCore.v2/02828884/31af3758c10b5b1274f1cdda9579594c/models/model_normalized.obj'

# # mesh = o3d.io.read_triangle_mesh(path2)
# # print(mesh)

# # mesh = o3d.io.read_triangle_mesh(path2)
# # print(mesh)

# mesh = trimesh.load(path2,force='mesh')
# vertices = np.asarray(mesh.vertices)
# print(vertices.shape)
# faces = np.asarray(mesh.faces)
# print(faces.shape)

# sample,_ = trimesh.sample.sample_surface(mesh, 1000)
# sample = np.asarray(sample)
# print(sample)
# # check if sample is numpy array:
# print(sample.shape)

# import os

# for root, _, files in os.walk('/root/autodl-tmp/ShapeNetCore.v2/'): 
#   for f in files:
#     if f.endswith(".png"):
#       path = os.path.join(root, f)
#       os.remove(path)


import torch

# # a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# # b = torch.tensor([1,2,2])
# # c = torch.gather(a,1,b.reshape(-1,1))
# # print(c)
# # # b as the index of the second dim of a
# p = torch.Tensor([[1,1,1],[2,2,2],[3,3,3]])
# a = torch.Tensor([
#     [1,2,3],[2,2,3],[5,4,1]
# ])
# x = torch.argmin(a,dim=1)
# print(p[x])


# Create a tensor of shape [32, 32, 32, 32, 3]
data = torch.randn(32, 32, 32, 32, 3)

# Create an index tensor of shape [32, 1000, 3]
indices = torch.randint(0, 32, size=(32, 1000, 3))

# Flatten the batch and index dimensions of the index tensor
flat_indices = indices.view(-1, 3)

# Compute the flat indices into the data tensor
flat_data_indices = (flat_indices[:, 0] * data.shape[1] * data.shape[2] * data.shape[3] +
                     flat_indices[:, 1] * data.shape[2] * data.shape[3] +
                     flat_indices[:, 2] * data.shape[3] +
                     torch.arange(data.shape[4]))

# Use the flat indices to gather the data
result = data.view(-1, data.shape[4])[flat_data_indices].view(32, 1000, 3)

print(result.shape)

