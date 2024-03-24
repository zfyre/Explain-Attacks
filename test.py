import torch

x = torch.rand([3, 32, 32], requires_grad=True)
y = torch.rand([3, 32, 32], requires_grad=False)

print(y)