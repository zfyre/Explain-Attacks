import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

# t = np.linspace(-10, 10, 1000)
# T = 10
# y = 1 - np.abs(t)/T

# def F(K):
#     x = np.empty_like(t)
#     x.fill(0)
#     for k in range(-K, K+1):
#         x += ((np.sinc(k/2)**2)/2) * (np.cos((np.pi*k*t)/T))
#     return x 


# plt.plot(F(100))
# plt.plot(y)

# plt.show()

# t = np.linspace(0, 10, 10)
# x = .5 ** t
# plt.plot(x)
# plt.show()


w = torch.atanh(2 * torch.rand([3, 32, 32]) - 1)
w = torch.nn.parameter.Parameter(w)


import torch.nn as nn

mse_loss = nn.MSELoss()
w = torch.randn(3, 4)  # Example tensor
loss = mse_loss(w, torch.zeros_like(w))
print(loss)  # Print the mean of the loss tensor