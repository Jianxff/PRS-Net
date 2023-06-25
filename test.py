
import torch
# a = torch.tensor([[1,2,3],[2,3,4],[4,5,6]])
# z = torch.zeros(a.shape[0],1)
# print('a',a)
# print('z',z)
# inputs = [a,z]
# # b = torch.cat(inputs,dim=1)

# b = torch.cat((torch.zeros(a.shape[0],1), a), dim=1)
# print(b)

# a = torch.Tensor([0,0,0])
# b = torch.Tensor([]).reshape(-1,3)
# c = torch.stack((a,b),dim=0)
# print(c)
a = torch.Tensor([1,2,3])
b = torch.Tensor([4,5,6])
c = torch.stack((a,b),dim=0)
print(c)