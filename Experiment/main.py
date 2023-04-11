import torch
import numpy as np
from Model import ISTA, FISTA, LISTA, LISTACP, TiLISTA
"""

"""


def main():
    path = './StimulateData/MatrixA.npy'
    A = np.load(path)
    A = torch.from_numpy(A)
    A = A.float()

    # A = torch.load('./StimulateData/MatrixB.pt')

    n, m = A.size(0), A.size(1)
    # print(n, m)

    x_single = torch.load('./StimulateData/x_single.pt')
    y_single = torch.load('./StimulateData/y_single.pt')
    # print(x_single)

    x_batch = torch.load('./StimulateData/x_batch.pt')
    y_batch = torch.load('./StimulateData/y_batch.pt')

    """
    ISTA
    """
    # ista_max_iteration = 6000
    # ista_Lasso_lamda = 0.125
    # ista_err = 1e-7
    # ista = ISTA.ISTA()
    # ista.ista(A, y_single, x_single, ista_max_iteration, ista_err, ista_Lasso_lamda)

    """
    FISTA
    """
    # fista_max_iteration = 6000
    # fista_Lasso_lamda = 0.125
    # fista_err = 1e-7
    # fista = FISTA.FISTA()
    # fista.fista(A, y_single, x_single, fista_max_iteration, fista_err, fista_Lasso_lamda)

    """
    LISTA
    """
    # lista_layer = 20
    # lista_Lasso_lambda = 0.6
    # lista_lr = 1e-2
    # lista = LISTA.train(x_batch, y_batch, A, lista_layer, lista_Lasso_lambda, lista_lr)

    """
    LISTA_CP
    """
    # listaCP_layer = 30
    # listaCP_Lasso_lambda = 0.25
    # listaCP_lr = 1e-3
    # listaCP = LISTACP.train(x_batch, y_batch, A, listaCP_layer, listaCP_Lasso_lambda, listaCP_lr)

    """
    TiLISTA
    """
    Tilista_layer = 16
    Tilista_Lasso_lambda = 0.25
    Tilista_lr = 1e-3
    Tilista = TiLISTA.train(x_batch, y_batch, A, Tilista_layer, Tilista_Lasso_lambda, Tilista_lr)





if __name__ == '__main__':
    main()
