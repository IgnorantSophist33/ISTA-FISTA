import torch
import numpy as np

path = './MatrixA.npy'
A = np.load(path)
A = torch.from_numpy(A)
A.float()

batch_size = 5000
probability = 0.1
n = A.size(0)
m = A.size(1)

x_batch = torch.zeros(1, m)
y_batch = torch.zeros(1, n)

for i in range(batch_size):
    x = torch.randn(m, 1)
    bernoulli = torch.distributions.bernoulli.Bernoulli(probability)
    bernoulli = bernoulli.sample((m, 1))
    x = np.multiply(x, bernoulli)
    epsilon = torch.randn((n, 1))
    y = torch.matmul(A, x) + epsilon
    x = x.view(1, m)
    y = y.view(1, n)
    if i == 0:
        x_batch = x
        y_batch = y
    else:
        x_batch = torch.cat([x_batch, x], dim=0)
        y_batch = torch.cat([y_batch, y], dim=0)


# print(x_batch.shape, y_batch.shape)
torch.save(x_batch, 'x_batch.pt')
torch.save(y_batch, 'y_batch.pt')