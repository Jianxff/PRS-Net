import torch

a = torch.Tensor([1,2,3])
b = a.unsqueeze(2)

print(b.shape)