#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018.3.27

@author:ningrun
'''
import numpy as np
from numpy import linalg as la
from scipy import sparse
import History
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# Solves the following problem via ADMM:
# minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
def lasso(A, b, lamda, rho, alpha):
    QUIET = False
    MAX_ITER = 1000
    ABSTOL = 10**(-4)
    RELTOL = 10**(-2)

    m, n = A.shape
    # save a matrix-vector multiply
    Atb = A.T*b

    # ADMM solver
    x = np.zeros([n, 1])
    z = np.zeros([n, 1])
    u = np.zeros([n, 1])

    # cache the factorization
    L, U = factor(A, rho)

    if QUIET is False:
        print('%3s' % 'iter', '%10s' % 'r norm', '%10s' % 'eps pri', '%10s' % 's norm', '%10s' % 'eps dual', '%10s' % 'objective')

    history = History.History()
    for k in range(0, MAX_ITER):
        # x-update
        q = Atb + rho*(z - u)  # temporary value
        if m >= n:
            x = la.solve(U.todense(), la.solve(L.todense(), q))
        else:
            x = q/rho - np.dot(A.T, la.solve(U.todense(), la.solve(L.todense(), np.dot(A, q))))/rho**2

        # z-update
        zold = z
        x_hat = alpha*x + (1 - alpha)*zold
        z = shrinkage(x_hat + u, lamda/rho)

        # u-update
        u = u + (x_hat - z)

        # diagnostics, reporting, termination checks
        history.addObjval(objective(A, b, lamda, x, z))
        history.addR_norm(la.norm(x - z))
        history.addS_norm(la.norm(-rho*(z - zold)))
        history.addEps_pri(np.sqrt(n)*ABSTOL + RELTOL*np.maximum(la.norm(x), la.norm(-z)))
        history.addEps_dual(np.sqrt(n)*ABSTOL + RELTOL*la.norm(rho*u))

        if QUIET is False:
            print('%3d' % k, '%10.4f' % history.getR_norm()[k], '%10.4f' % history.getEps_pri()[k], '%10.4f' % history.getS_norm()[k],
                  '%10.4f' % history.getEps_dual()[k], '%10.2f' % history.getObjval()[k])

        if history.getR_norm()[k] < history.getEps_pri()[k] and history.getS_norm()[k]<history.getEps_dual()[k]:
            break
    return z, history


def objective(A, b, lamda, x, z):
    a = np.dot(A, x) - b
    p = 1/2*sum((np.multiply(a, a))) + lamda*la.norm(z, 1)
    return p[0, 0]


def shrinkage(x, kappa):
    z = np.maximum(0, (x - kappa).getA()) - np.maximum(0, (-x - kappa).getA())
    return z


def factor(A, rho):
    m, n = A.shape
    if m >= n:
        L = la.cholesky(np.dot(A.T, A) + rho*sparse.eye(n))
    else:
        L = la.cholesky(sparse.eye(m) + 1/rho*(np.dot(A, A.T)))

    L = sparse.coo_matrix(L)
    U = sparse.coo_matrix(L.T)

    return L, U







