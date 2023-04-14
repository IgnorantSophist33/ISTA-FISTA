import torch
import numpy as np
import torch.nn.functional as F
from scipy.linalg import orth

n = 250
m = 500
A = np.random.normal(0, 1/n, (n, m))
B = torch.from_numpy(A)
B = B.float()
print(B.type, B.shape)
print(B)
B = F.normalize(B, dim=0, p=2)



torch.save(B, 'MatrixB.pt')