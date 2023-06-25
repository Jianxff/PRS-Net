
import torch.tensor as tensor

# a = torch.tensor([[1,2,3],[2,3,4],[4,5,6]])
# z = torch.zeros(a.shape[0],1)
# print('a',a)
# print('z',z)
# inputs = [a,z]
# # b = torch.cat(inputs,dim=1)

# b = torch.cat((torch.zeros(a.shape[0],1), a), dim=1)
# print(b)

quat = tensor([1,2,3,4])
quat_inv = quat * tensor([1,-1,-1,-1])
print(quat_inv)