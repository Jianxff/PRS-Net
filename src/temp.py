import torch

a = torch.Tensor([[1,2,3],[1,2,2],[2,3,1]])
b = torch.norm(a, dim=1)
c = torch.norm(a, dim=1)**2

print('b:',b)
print('c:',c)