import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim
from visdom import Visdom
"""
A : measurement matrix
x : spare vector
x_hat: iteration variable
y : measurement vector
max_iteration : max iteration times
Lasso_lambda : regularization parameter 
L : the Lipschitz constant
max_iteration : the layers of the net
"""

class LISTA_CP(nn.Module):
    def __init__(self, A, max_iteration, Lasso_lambda):
        super(LISTA_CP, self).__init__()
        self.A = A
        self.n = A.size(0)
        self.m = A.size(1)
        self.max_iteration = max_iteration
        self.Lasso_lambda = Lasso_lambda
        self.L = self.Maxeigenvalue(A)
        self.theta = nn.Parameter(torch.tensor([Lasso_lambda/self.L ]).repeat(self.max_iteration))
        # self.W_x = nn.Linear(in_features=self.m, out_features=self.m, bias=False)
        # self.W_y = nn.Linear(in_features=self.n, out_features=self.m, bias=False)
        # self.W_x = nn.Parameter(torch.zeros((max_iteration, self.m, self.m)))
        # self.W_y = nn.Parameter(torch.zeros((max_iteration, self.n, self.m)))
        self.W = nn.Parameter(torch.zeros((max_iteration, self.n, self.m)))
        # self.shrinkage = nn.Softshrink(1e-2)

    def Maxeigenvalue(self, A):
        eig, eig_vector = np.linalg.eig(torch.matmul(A.T, A))
        return np.max(eig).real

    def weights_initialise(self):
        x = torch.eye(self.A.size(1)) - (1/self.L) * torch.matmul(self.A.T, self.A)
        y = (1/self.L) * self.A.T
        # self.W_x = nn.Parameter(x.T.unsqueeze(0).repeat(self.max_iteration, 1, 1))
        self.W = nn.Parameter(y.T.unsqueeze(0).repeat(self.max_iteration, 1, 1))
        # print(self.W_y.shape, self.W_x.shape)

    def forward(self, y):
        shrinkage = nn.Softshrink(torch.abs(self.theta[0]).item())
        x_hat = shrinkage(torch.matmul(y, self.W[0]))

        # theta = self.theta * torch.ones(self.m, y.size(1))
        for i in range(self.max_iteration-1):
            shrinkage = nn.Softshrink(torch.abs(self.theta[i+1]).item())
            temp = torch.matmul(x_hat, self.A.T)
            x_hat = shrinkage(x_hat + torch.matmul(y - temp, self.W[i+1]))
        return x_hat

def NMSEdB(x, x_hat):
    # x = x.unsqueeze(1)
    # x_hat = x_hat.unsqueeze(1)
    vec_temp1 = x - x_hat
    vec_temp2 = x
    norm1 = torch.pow(torch.norm(vec_temp1, p=2, dim=1), 2)
    norm2 = torch.pow(torch.norm(vec_temp2, p=2, dim=1), 2)
    # print(norm2.shape)
    result = torch.sum(10 * torch.log10(norm1 / norm2))/x.size(0)
    return result

def train(x, y, A, max_iteration, Lasso_lambda, lr):
        viz = Visdom()
        batch_size = 32
        n_samples = y.size(0) - batch_size

        steps = n_samples // batch_size

        x_test = x[x.size(0) - batch_size:x.size(0), :]
        y_test = y[x.size(0) - batch_size:x.size(0), :]
        x_comp = x[0:batch_size, :]
        y_comp = y[0:batch_size, :]

        lista_CP = LISTA_CP(A, max_iteration, Lasso_lambda)
        lista_CP.weights_initialise()

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(lista_CP.parameters(), lr=lr, momentum=0.9)

        viz.line(Y=np.column_stack([0., 0.]), X=np.column_stack([0, 0]), win='NMSE',
                 opts=dict(legend=["test", "train"],
                           title='NMSE',
                           xlabel='times',
                           ylabel='NMSE'))
        i = 0
        for epoch in range(1000):
            index_samples = np.random.choice(a=n_samples, size=n_samples, replace=False, p=None)
            y_shuffle = y[index_samples]
            x_shuffle = x[index_samples]
            for step in range(steps):
                optimizer.zero_grad()
                i = i + 1
                y_batch = y_shuffle[step * batch_size:(step+1) * batch_size]
                x_batch = x_shuffle[step * batch_size:(step+1) * batch_size]
                # print(y_batch.shape)
                x_hat = lista_CP(y_batch)


                loss = criterion(x_batch, x_hat)
                loss.backward()
                optimizer.step()

                # print("W_x", lista.W_x)
                # print("W_y", lista.W_y)
                # print("theta: ", lista.theta)



                x_test_hat = lista_CP(y_test)
                x_comp_hat = lista_CP(y_comp)
                nmse1 = NMSEdB(x_test, x_test_hat)
                nmse2 = NMSEdB(x_comp, x_comp_hat)
                viz.line(Y=np.column_stack((nmse1.item(), nmse2.item())), X=np.column_stack((i, i)), win='NMSE',
                         update='append',
                         opts=dict(legend=["test", "train"],
                                   title='NMSE',
                                   xlabel='times',
                                   ylabel='NMSE'))


        return lista_CP
