#!/usr/bin/python
# coding:utf-8
# 参照URL: https://github.com/hagino3000/public-ipynb/blob/master/PRML/PRML%203.3.ipynb

from __future__ import division
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
#import data

ALPHA = 2.0
BETA = 25
I = np.identity(2)
C = [-0.3, 0.5]
DATA_SIZE = 100

"""
xから計画行列Phiを作る関数
"""
def design_matrix(x):
    return np.array([[1, xi] for xi in x])
"""
mn = BETA*Sn*(Phi.T)*tを求める関数
"""
def calc_mn(x, t):
    Phi = design_matrix(x)
    Sn = calc_Sn(x)
    return BETA*Sn.dot(Phi.T).dot(t)

# Sn = (a*I + BETA*(Phi.T)*Phi).invを求める関数
def calc_Sn(x):
    Phi = design_matrix(x)
    return np.linalg.inv(ALPHA*I + BETA*Phi.T.dot(Phi))

"""
プロット作成関数
"""
def plot_probability(mean = [0,0], cov = I, title = '', ax = None):
    w0 = np.linspace(-1, 1, 100)
    w1 = np.linspace(-1, 1, 100)
    W0, W1 = np.meshgrid(w0, w1)
    P = []
    for w0i in w0:
        P.append([multivariate_normal.pdf([w0i, w1i], mean, cov) for w1i in w1])
    ax.pcolor(W0, W1, np.array(P).T, cmap = plt.cm.jet)
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_1$')
    ax.set_title(title)

"""
観測値1つの対数尤度を求める
"""
def calc_likelihood(t, x, w):
    w = np.array(w)
    phi_x = np.array([1,x])
    return -1 * BETA / 2 * (t - w.T.dot(phi_x))**2

"""
観測値の尤度のプロット
"""
def plot_likelihood(t, x, title = '', ax = None):
    w0 = np.linspace(-1, 1, 200)
    w1 = np.linspace(-1, 1, 200)
    W0, W1 = np.meshgrid(w0, w1)
    L = []
    for w0i in w0:
        L.append([calc_likelihood(t, x, [w0i, w1i]) for w1i in w1])
    ax.pcolor(W0, W1, np.array(L).T, cmap = plt.cm.jet, vmax = 0, vmin = -1)
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_1$')
    ax.set_title(title)

def plot_result_line(w, ax):
    x = np.linspace(-1, 1, 100)
    y = w[0] + x*w[1]
    line, = ax.plot(x, y)
    return line

def plot_lines(mean, cov, ax):
    plot_result_line(np.random.multivariate_normal(mean, cov), ax)
    plot_result_line(np.random.multivariate_normal(mean, cov), ax)
    plot_result_line(np.random.multivariate_normal(mean, cov), ax)
    plot_result_line(np.random.multivariate_normal(mean, cov), ax)
    plot_result_line(np.random.multivariate_normal(mean, cov), ax)
    for (x, t) in zip(train_x, train_t):
        ax.plot(x, t, 'ro')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(u'Data Space')

"""
データ点の生成
"""
def create_data():
    def f(x):
        return C[0] + C[1]*x + np.random.normal(0, 0.2)
    x = (np.random.random() - 0.5) * 2
    t = f(x)
    return x, t

"""
逐次学習のupdate
"""
def update(train_x, train_t):
    x, t = create_data()
    train_x.append(x)
    train_t.append(t)
    mn = calc_mn(x = train_x, t = train_t)
    sn = calc_Sn(x = train_x)
    return mn, sn

def create_fig3_7():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (11,3))
    ax1.grid(True)
    ax3.grid(True)
    plt.subplots_adjust(wspace=0.4)
    m0 = [0, 0]
    s0 = ALPHA * I
    plot_probability(mean = m0, cov = s0, title = u'Prior Probability $p(w1\\alpha)$', ax = ax2)
    plot_lines(m0, s0, ax = ax3)

    def init_animation():
        print 'init animation'

    def update_animation(frame):
        print 'Fig 3.7 - sample : %d' % frame
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax3.grid(True)
        update(train_x, train_t)
        mn = calc_mn(x = train_x, t = train_t)
        sn = calc_Sn(x = train_x)
        plt.subplots_adjust(wspace=0.4)
        plot_likelihood(t = train_t[-1], x = train_x[-1], title = u'%d Likelihood' % (frame + 1), ax = ax1)
        plot_probability(mean = mn, cov = sn, title = u'Posterior probability$p(w|t), n = %d$' % (frame + 1), ax = ax2)
        plot_lines(mn, sn, ax = ax3)
        ax3.set_title(u'Data Space')
        return frame
    print 'Create Fig 3.7 ............'
    ani = animation.FuncAnimation(fig, update_animation, frames = 100, blit=False, init_func = init_animation, repeat = False)
    ani.save('fig3_7_baysian_linear_regression_alpha%f_beta%d.gif'%(ALPHA, BETA), writer='imagemagick', fps=3)
    #ani = animation.FuncAnimation(fig, update_animation, frames = 100, blit=False, init_func = init_animation, repeat = False, interval = 300)
    #plt.show()

if __name__ == '__main__':
    train_x = []
    train_t = []

    create_fig3_7()
