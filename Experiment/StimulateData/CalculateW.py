import numpy as np
import torch

def func(W, A):
    m = A.size(1)
    res = torch.matmul(A.T, W) - torch.eye(m)
    Q = torch.ones(m, m) + torch.eye(m) * -1
    temp = torch.from_numpy(np.sqrt(Q.numpy()))
    res = res * temp
    f = torch.sum(res * res)
    return f

def proj(W, A):
    aw = torch.diag(torch.matmul(A.T, W))
    # aw = torch.diag_embed(aw)
    aw = aw.repeat(A.size(0), 1)
    W_next = W + (1 - aw) * A
    return W_next

path = './MatrixA.npy'
# A = np.load(path)
# A = torch.from_numpy(A)
# A = A.float()

A = torch.load('./MatrixB.pt')

n, m = A.size(0), A.size(1)
W = A
f = func(A, A)
step_size = 0.1

for i in range(2000):
    res = torch.matmul(A.T, W) - torch.eye(m)
    gra = torch.matmul(A, res)

    W_next = W - step_size * gra
    W_next = proj(W_next, A)
    f_next = func(W_next, A)

    if torch.abs(f - f_next) / f < 1e-12: break

    W = W_next
    f = f_next

    if i % 50 == 0:
        print('i:{}, func:{}'.format(i, f))

print(W)
torch.save(W, 'MatrixW.pt')