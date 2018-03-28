#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018.3.27

@author:ningrun
'''
import numpy as np
from numpy import linalg as la
from scipy import sparse
from scipy import stats
import matplotlib.pyplot as plt
import Lasso

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    # Generate problem data
    m = 1500  # number of examples
    n = 5000  # number of features
    p = np.true_divide(100, n)  # sparsity density

    # 稀疏正态分布随机矩阵
    x0 = np.random.randn(int(n * p), 1)
    z = np.zeros([n - len(x0), 1])
    x0 = np.vstack((x0, z))
    np.random.shuffle(x0)
    x0 = np.mat(x0)

    # 二维正态分布
    A = np.random.randn(m, n)
    A = np.mat(A)

    A = np.dot(A, sparse.spdiags(1/np.sqrt(sum(np.multiply(A, A))), 0, n, n).todense())  # normalize column
    b = np.dot(A, x0) + np.sqrt(0.001)*np.random.randn(m, 1)

    lambda_max = la.norm(np.dot(A.T, b), np.inf)
    lamda = 0.1*lambda_max

    # Solve problem
    x, history = Lasso.lasso(A, b, lamda, 1.0, 1.0)

    # Reporting
    K = len(history.getObjval())
    x = np.arange(K)
    plt.plot(x, history.getObjval())
    plt.xlabel('iter (k)')
    plt.ylabel('f(x^k) + g(z^k)')
    plt.show()

    plt.subplot(211)
    plt.plot(x, np.maximum(10**(-8), history.getR_norm()), '-', history.getEps_pri(), '--')
    plt.ylabel('||r||_2')

    plt.subplot(212)
    plt.plot(x, np.maximum(10**(-8), history.getS_norm()), '-', history.getEps_dual(), '--')
    plt.xlabel('iter (k)')
    plt.ylabel('||s||_2')
    plt.show()










