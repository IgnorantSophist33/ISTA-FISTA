import numpy as np
import torch

path = './MatrixA.npy'
A = np.load(path)
A = torch.from_numpy(A)
A = A.float()

# A = torch.load('./MatrixB.pt')

probability = 0.1
n = A.size(0)
m = A.size(1)

x = torch.randn(m, 1)
bernoulli = torch.distributions.bernoulli.Bernoulli(probability)
bernoulli = bernoulli.sample((m, 1))
x = np.multiply(x, bernoulli)
# print(x.shape, x)
epsilon = torch.randn((n, 1))
y = torch.matmul(A, x) + epsilon
# print(y, y.shape)

torch.save(x, 'x_single.pt')
torch.save(y, 'y_single.pt')