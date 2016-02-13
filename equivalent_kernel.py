#!/usr/bin/python
# coding:utf-8
# 参照URL: 

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

ALPHA = 2.0
BETA = 25
S = 0.1
X_MIN = -1.
X_MAX = 1.
DATA_SIZE = 200
BASIS_SIZE = 9.
I = np.identity(BASIS_SIZE)


def phi(x_n):
    return np.array([basis(x_n) for basis in bases])

def design_matrix(x, bases):
    return np.array(map(phi, x))

def gaussian_basis_func(mu_j, s):
    return lambda x:np.exp(-(x - mu_j)**2 / (2. * s**2.))

def gaussian_basis_funcs(mu, s):
    return [gaussian_basis_func(mu_j, s) for mu_j in mu]

def gaussian_basis_func(mu_j, s):
    return lambda x:np.exp(-(x - mu_j)**2 / (2. * s**2.))

def equivalent_kernel(x, x_prime, S_n):
    phi_x = np.array(map(phi,x)).T
    phi_x_prime = np.array(map(phi,x_prime)).T
    return BETA * (phi_x.T).dot(S_n).dot(phi_x_prime)


if __name__ == '__main__':
    x = np.arange(X_MIN, X_MAX, (X_MAX - X_MIN) / DATA_SIZE)
    x_prime = np.arange(X_MIN, X_MAX, (X_MAX - X_MIN) / DATA_SIZE)

    mu = np.arange(X_MIN, X_MAX + 0.01, (X_MAX - X_MIN) / (BASIS_SIZE - 1.))
    bases = gaussian_basis_funcs(mu, S) # 基底関数はガウス基底を採用
    Phi = design_matrix(x, bases)
    S_N_inv = ALPHA*I + BETA*Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = BETA * S_N.dot(Phi.T).dot(x)


    fig, ax = plt.subplots(1, 1, figsize = (11,11))
    fig.suptitle('fig. 3.10', fontsize = 14, fontweight = 'bold')
    plt.subplots_adjust(wspace=0.4)
    ax.pcolor(x, x_prime, equivalent_kernel(x, x_prime, S_N), cmap = plt.cm.jet)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$x\'$')
    ax.set_title('Equivalent Kernel')
    plt.savefig('fig3_10_equivalent_kernel.png')
    plt.show()
