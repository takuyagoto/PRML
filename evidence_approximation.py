#!/usr/bin/python
# coding:utf-8
# 参照URL: http://qiita.com/naoya_t/items/80ea108cebc694f5cd63

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

S = 0.1
ALPHA = 0.
BETA = 1.
DATA_SIZE = 50
BASIS_SIZE = 9.

I = np.eye(BASIS_SIZE)

"""
ガウス基底関数φj(x)
"""
def gaussian_basis_func(mu_j, s):
    return lambda x:np.exp(-(x - mu_j)**2 / (2. * s**2.))

"""
基底ベクトルφ(x)
"""
def gaussian_basis_funcs(mu, s):
    return [gaussian_basis_func(mu_j, s) for mu_j in mu]

def phi(x_n):
    return np.array([basis(x_n) for basis in bases])

"""
計画行列Φ(x)
"""
def design_matrix(x, bases):
    return np.array(map(phi, x))

"""
予測分布
"""
def predictive_dist_func(alpha, beta, x, t, bases):
    Phi = design_matrix(x, bases)
    S_N_inv = alpha*I + beta*Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    def func(x_n):
        phi_x_n = phi(x_n)
        mu = (m_N.T).dot(phi_x_n)
        s2_N = 1. / beta + (phi_x_n.T).dot(S_N).dot(phi_x_n)
        return (mu, s2_N)

    return m_N, S_N, func

def update_hyper_param(m_N, S_N, x, t):
    E_w = 0.5*(m_N.T).dot(m_N)
    E_D = sum([0.5*(t_n - (m_N.T).dot(phi(x_n)))**2 for x_n, t_n in zip(x, t)])
    alpha = BASIS_SIZE / (2 * E_w)
    beta = DATA_SIZE / (2 * E_D)
    return alpha, beta

"""
データ点の生成
"""
def create_data():
    def f(x):
        return np.sin(x*np.pi*2) + np.random.normal(0, 0.2)
    x = np.array([np.random.random() for i in xrange(DATA_SIZE)])
    t = np.array([f(x_n) for x_n in x])
    return x, t


def create_fig():
    train_x, train_t = create_data()
    mu = np.arange(0, 1.01, 1. / (BASIS_SIZE - 1.))
    bases = gaussian_basis_funcs(mu, S)
    _x = np.arange(0, 1, 0.01)
    _alpha = [ALPHA]
    _beta = [BETA]

    xmin = -0.05
    xmax = 1.05
    ymin = -1.5
    ymax = 1.5

    fig, ax = plt.subplots(1, 1, figsize = (11,11))
    fig.suptitle('Evidence Approximation (%d samples)' % DATA_SIZE, fontsize = 14, fontweight = 'bold')
    plt.subplots_adjust(wspace=0.4)

    ax.set_title(u'ep. 0')
    ax.grid(True)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.plot(_x, np.sin(_x*np.pi*2), color = 'r')
    for data_x, data_t in zip(train_x, train_t):
        ax.scatter(data_x, data_t, color = 'b', marker = 'o')

    m_N, S_N, f = predictive_dist_func(_alpha[-1], _beta[-1], train_x, train_t, bases)

    y = []
    for mu, s2 in map(f, _x):
        y.append(mu)

    line, = ax.plot(_x, y, color = '#000000')
    info_text = ax.text(0.8, 1., (r'$\alpha = %f$' + '\n' + r'$\beta = %f$') % (_alpha[-1], _beta[-1]))
    a, b = update_hyper_param(m_N, S_N, train_x, train_t)
    _alpha.append(a)
    _beta.append(b)

    def init():
        print "init animation"

    def update(frame):
        print  'Fig epsode : %d' % (frame + 1)
        ax.set_title(u'ep. %d' % (frame + 1))
        info_text.set_text((r'$\alpha = %f$' + '\n' + r'$\beta = %f$') % (_alpha[-1], _beta[-1]))
        m_N, S_N, f = predictive_dist_func(_alpha[-1], _beta[-1], train_x, train_t, bases)
        y = []
        for mu, s2 in map(f, _x):
            y.append(mu)
        line.set_ydata(y)
        a, b = update_hyper_param(m_N, S_N, train_x, train_t)
        _alpha.append(a)
        _beta.append(b)
        return frame

    print 'Create fig .....'
    ani = animation.FuncAnimation(fig, update, frames = 30, blit=False, repeat = False, init_func = init)
    ani.save('fig_evidence_approximation_data_%d.gif'%DATA_SIZE, writer='imagemagick', fps=3)
    #ani = animation.FuncAnimation(fig, update, frames = 50, blit=False, repeat = False, interval = 500)
    #plt.show()

if __name__ == '__main__':
    mu = np.arange(0, 1.01, 1. / (BASIS_SIZE - 1.))
    bases = gaussian_basis_funcs(mu, S)
    _alpha = ALPHA
    _beta = BETA
    create_fig()
