#!/usr/bin/python
# coding:utf-8
# 参照URL: http://qiita.com/naoya_t/items/80ea108cebc694f5cd63

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

S = 0.1
ALPHA = 0.1
BETA = 9.
DATA_SIZE = 10.
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

"""
データ点の生成
"""
def create_data():
    def f(x):
        return np.sin(x*np.pi*2) + np.random.normal(0, 0.2)
    x = np.random.random()
    t = f(x)
    return x, t


def create_fig3_8():
    train_x = []
    train_t = []
    mu = np.arange(0, 1.01, 1. / (BASIS_SIZE - 1.))
    bases = gaussian_basis_funcs(mu, S)
    _x = np.arange(0, 1, 0.01)

    xmin = -0.05
    xmax = 1.05
    ymin = -1.5
    ymax = 1.5

    fig, ax = plt.subplots(1, 1, figsize = (11,11))
    fig.suptitle('fig. 3.8', fontsize = 14, fontweight = 'bold')
    plt.subplots_adjust(wspace=0.4)

    ax.grid(True)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.plot(_x, np.sin(_x*np.pi*2), color = 'r')

    x, t = create_data()
    train_x.append(x)
    train_t.append(t)
    m_N, S_N, f = predictive_dist_func(ALPHA, BETA, train_x, train_t, bases)

    y_h = []
    y_m = []
    y_l = []
    for mu, s2 in map(f, _x):
        s = np.sqrt(s2)
        y_h.append(mu + s)
        y_m.append(mu)
        y_l.append(mu - s)

    fill = ax.fill_between(_x, y_h, y_l, color = '#cccccc')
    line, = ax.plot(_x, y_m, color = '#000000')
    ax.scatter(train_x[-1], train_t[-1], color = 'b', marker = 'o')

    def init():
        print "init animation"

    def update(frame):
        print  'Fig 3.8 - sample : %d' % frame
        ax.set_title(u'%d samples' % (frame + 1))
        x, t = create_data()
        train_x.append(x)
        train_t.append(t)
        m_N, S_N, f = predictive_dist_func(ALPHA, BETA, train_x, train_t, bases)
        y_h = []
        y_m = []
        y_l = []
        for mu, s2 in map(f, _x):
            s = np.sqrt(s2)
            y_h.append(mu + s)
            y_m.append(mu)
            y_l.append(mu - s)
        path = fill.get_paths()[0]
        new_data = [y_l[0]] 
        new_data += y_h + [y_l[-1]]
        y_l.reverse()
        new_data += y_l + [y_l[-1]]
        path.vertices[:, 1] = new_data

        line.set_ydata(y_m)
        ax.scatter(train_x[-1], train_t[-1], color = 'b', marker = 'o')

        return frame

    print 'Create fig. 3.8 .....'
    ani = animation.FuncAnimation(fig, update, frames = 100, blit=False, repeat = False, init_func = init)
    strAlpha = '%f' % ALPHA
    strAlpha = strAlpha.replace('0.', '')
    ani.save('fig3_8_predictive_distribution%s_beta%d.gif'%(strAlpha, BETA), writer='imagemagick', fps=3)
    #ani = animation.FuncAnimation(fig, update, frames = 50, blit=False, repeat = False, interval = 500)
    #plt.show()

def create_fig3_9():
    train_x = []
    train_t = []
    _x = np.arange(0, 1, 0.01)

    xmin = -0.05
    xmax = 1.05
    ymin = -1.5
    ymax = 1.5

    fig, ax = plt.subplots(1, 1, figsize = (11,11))
    fig.suptitle('fig. 3.9', fontsize = 14, fontweight = 'bold')
    plt.subplots_adjust(wspace=0.4)

    ax.grid(True)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.plot(_x, np.sin(_x*np.pi*2), color = 'r')

    x, t = create_data()
    train_x.append(x)
    train_t.append(t)
    m_N, S_N, f = predictive_dist_func(ALPHA, BETA, train_x, train_t, bases)

    lines = []
    for i in range(5):
        w = (np.random.multivariate_normal(m_N, S_N, 1)).T
        y = lambda x: (w.T).dot(phi(x))[0]
        line, = ax.plot(_x, y(_x), color = '#cccccc')
        lines.append(line)

    ax.scatter(train_x[-1], train_t[-1], color = 'b', marker = 'o')

    def init():
        print "init animation"

    def update(frame):
        print  'Fig 3.9 - sample : %d' % frame
        ax.set_title(u'%d samples' % (frame + 1))
        x, t = create_data()
        train_x.append(x)
        train_t.append(t)
        m_N, S_N, f = predictive_dist_func(ALPHA, BETA, train_x, train_t, bases)
        for line in lines:
            w = (np.random.multivariate_normal(m_N, S_N, 1)).T
            y = lambda x: (w.T).dot(phi(x))[0]
            line.set_ydata(y(_x))
        ax.scatter(train_x[-1], train_t[-1], color = 'b', marker = 'o')
        return frame

    print 'Create fig. 3.9 .....'
    ani = animation.FuncAnimation(fig, update, frames = 100, blit=False, repeat = False, init_func = init)
    strAlpha = '%f' % ALPHA
    strAlpha = strAlpha.replace('0.', '')
    ani.save('fig3_9_predictive_distribution%s_beta%d.gif'%(strAlpha, BETA), writer='imagemagick', fps=3)
    #ani = animation.FuncAnimation(fig, update, frames = 50, blit=False, repeat = False, interval = 500, init_func = init)
    #plt.show()


if __name__ == '__main__':
    mu = np.arange(0, 1.01, 1. / (BASIS_SIZE - 1.))
    bases = gaussian_basis_funcs(mu, S)
    create_fig3_8()
    create_fig3_9()
