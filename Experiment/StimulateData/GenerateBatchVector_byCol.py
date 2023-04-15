import torch
import numpy as np

path = './MatrixA.npy'
A = np.load(path)
A = torch.from_numpy(A)
A.float()

# A = torch.load('./MatrixB.pt')

batch_size = 2000
probability = 0.1
n = A.size(0)
m = A.size(1)

x_batch = torch.zeros(m, 1)
y_batch = torch.zeros(n, 1)

for i in range(batch_size):
    x = torch.randn(m, 1)
    bernoulli = torch.distributions.bernoulli.Bernoulli(probability)
    bernoulli = bernoulli.sample((m, 1))
    x = np.multiply(x, bernoulli)
    epsilon = torch.randn((n, 1))
    y = torch.matmul(A, x) + epsilon
    x = x.view(m, 1)
    y = y.view(n, 1)
    if i == 0:
        x_batch = x
        y_batch = y
    else:
        x_batch = torch.cat([x_batch, x], dim=1)
        y_batch = torch.cat([y_batch, y], dim=1)


print(x_batch.shape, y_batch.shape)
torch.save(x_batch, 'x_batch_col.pt')
torch.save(y_batch, 'y_batch_col.pt')