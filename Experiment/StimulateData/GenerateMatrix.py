import torch
import numpy as np
import torch.nn.functional as F
from scipy.linalg import orth

n = 500
m = 1000
# A = np.random.normal(0, 1/n, (n, m))
# B = torch.from_numpy(A)
# B = B.float()
# print(B.type, B.shape)
# print(B)
# B = F.normalize(B, dim=0, p=2)

Psi = np.eye(m)
Phi = np.random.randn(n, m)
Phi = np.transpose(orth(np.transpose(Phi)))
B = np.dot(Phi, Psi)
print(B.shape)
print(B)
B = torch.from_numpy(B)
B = B.float()
B = F.normalize(B, dim=1, p=2)

torch.save(B, 'MatrixB.pt')