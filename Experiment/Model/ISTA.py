import torch
import  numpy as np
from visdom import Visdom
import time

"""
A : measurement matrix
x : spare vector
x_hat, x_hat_new : iteration variable
y : measurement vector
max_iteration : max iteration times
err : stopping criteria
Lasso_lambda : regularization parameter 
L : the Lipschitz constant
"""

class ISTA():
    def __init__(self):
        super(ISTA, self).__init__()

    def timer(self, criterion, criterion_val, step, timetable, t):
        if criterion < criterion_val:
            t1 = time.time()
            criterion_val = criterion_val + step
            run_time = t1 - t
            timetable.append(round(run_time, 3))
        return criterion_val

    def shrinkage(self, x, theta):
        temp = torch.cat([torch.abs(x) - theta, torch.zeros(x.size(0), 1)], dim=1)
        max_val, max_idx = torch.max(temp, 1)
        max_val = max_val.unsqueeze(1)
        max_nd = max_val.numpy()
        return np.multiply(np.sign(x), max_nd)

    def Maxeigenvalue(self, A):
        eig, eig_vector = np.linalg.eig(torch.matmul(A.T, A))
        return np.max(eig).real

    def NMSEdB(self, x, x_hat):
        vec_temp1 = x - x_hat
        vec_temp2 = x
        norm1 = torch.pow(torch.norm(vec_temp1, p=2), 2)
        norm2 = torch.pow(torch.norm(vec_temp2, p=2), 2)
        result = 10 * torch.log10(norm1 / norm2)
        return  result

    def ista(self, A, y, x, max_iteration, err, Lasso_lambda):
        L = self.Maxeigenvalue(A)+1
        A_L = (1/L) * A.T

        x_hat = torch.zeros((A.shape[1], 1))
        viz = Visdom()
        viz.line([0.], [0], win="ISTA_NMSEdB", opts=dict(title='ISTA_NMSEdB'))
        viz.line([0.], [0], win="ISTA_L2^2ERROR", opts=dict(title='ISTA_L2^2ERROR'))
        viz.line([0.], [0], win="ISTA_L1ERROR", opts=dict(title='ISTA_L1ERROR'))

        t = time.time()
        timetable = []
        criterion_val = 0.
        step = -0.5
        nmse, L2err, L1err = 0, 0, 0
        for i in range(max_iteration):
            temp = y - torch.matmul(A, x_hat)
            x_hat_new = self.shrinkage(x_hat + torch.matmul(A_L, temp), Lasso_lambda / L)
            if torch.abs(x_hat_new - x_hat).sum() <= err:
                print("already converged")
                break
            x_hat = x_hat_new

            nmse = self.NMSEdB(x, x_hat)
            L2err = torch.pow(torch.norm(x - x_hat, p=2), 2)/x.size(0)
            L1err = torch.norm(x - x_hat, p=1)/x.size(0)
            viz.line([nmse], [i], win='ISTA_NMSEdB', update='append')
            viz.line([L1err], [i], win='ISTA_L1ERROR', update='append')
            viz.line([L2err], [i], win='ISTA_L2^2ERROR', update='append')
            # print('iter:{}, nmsedB:{}, L1:{}, L2:{}'.format(i, nmse, L1err, L2err))

            criterion_val = self.timer(nmse, criterion_val, step, timetable, t)

        print("ISTA-NMSE: ", timetable)
        print('iter:{}, nmsedB:{}, L1:{}, L2:{}'.format(max_iteration, nmse, L1err, L2err))
        # path = '../Result/istatime.txt'
        # file = open(path, 'w')
        # file.write(timetable)
        return  x_hat





