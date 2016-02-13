#!/usr/bin/python
#coding:utf-8

import numpy as np
import math
from matplotlib import animation
from matplotlib import pyplot as plt
from pylab import maximum
from pylab import minimum


DATA_SIZE = 20
OUTLIER_SIZE = 10

"""
入力にダミー入力1を付け足したX~を作成
"""
def create_X_tilde(X):
    return np.array([np.r_[1,xi] for xi in X])

"""
正規方程式の解
"""
def calc_weight(X, T):
    X_tilde = create_X_tilde(X)
    W_tilde = np.linalg.inv(X_tilde.T.dot(X_tilde)).dot(X_tilde.T).dot(T)
    return W_tilde

def get_border(W):
    def func(x):
        return (-x * (W[1][0] - W[1][1]) - (W[0][0] - W[0][1]))/(W[2][0] - W[2][1])
    return func

def create_fig4_4():
    def create_data1():
        def f(x):
            return np.random.normal(0, 0.4) + x + 2.
        x1 = 6. * np.random.random_sample(DATA_SIZE,) - 4.
        x2 = np.array(map(f, x1))
        t = np.array([[1, 0] for i in xrange(DATA_SIZE)])
        return np.array(zip(x1, x2)), t

    def create_data2():
        def f(x):
            return np.random.normal(0, 0.4) + x - 2.
        x1 = 6. * np.random.random_sample(DATA_SIZE,) - 2.
        x2 = np.array(map(f, x1))
        t = np.array([[0, 1] for i in xrange(DATA_SIZE)])
        return np.array(zip(x1, x2)), t

    def create_outlier():
        x1 = np.array([np.random.normal(8., 0.8) for i in xrange(OUTLIER_SIZE)])
        x2 = np.array([np.random.normal(-6., 0.8) for i in xrange(OUTLIER_SIZE)])
        t = np.array([[0, 1] for i in xrange(OUTLIER_SIZE)])
        return np.array(zip(x1, x2)), t

    X1, T1 = create_data1()
    X2, T2 = create_data2()
    X3, T3 = create_outlier()

    W1 = calc_weight(np.r_[X1, X2], np.r_[T1, T2])
    W2 = calc_weight(np.r_[X1, X2, X3], np.r_[T1, T2, T3])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))
    ax1.grid(True)
    ax2.grid(True)
    plt.subplots_adjust(wspace = 0.4)
    ax1.set_xlim(-4, 9)
    ax1.set_ylim(-9, 4)
    ax2.set_xlim(-4, 9)
    ax2.set_ylim(-9, 4)

    ax1.scatter(X1[:,0], X1[:,1], s = 50, c = 'r', marker = "x")
    ax1.scatter(X2[:,0], X2[:,1], s = 50, edgecolors = 'b', marker = "o", facecolors= 'none')

    ax2.scatter(X1[:,0], X1[:,1], s = 50, c = 'r', marker = "x")
    ax2.scatter(X2[:,0], X2[:,1], s = 50, edgecolors = 'b', marker = "o", facecolors= 'none')
    ax2.scatter(X3[:,0], X3[:,1], s = 50, edgecolors = 'b', marker = "o", facecolors= 'none')

    x = np.arange(-10, 10, 0.1)
    border_func1 = get_border(W1)
    ax1.plot(x, map(border_func1, x), 'm')

    border_func2 = get_border(W2)
    ax2.plot(x, map(border_func2, x), 'm')

    plt.show()
    #plt.save_fig('fig4_4.png')

def create_fig4_5():
    def create_data1():
        def f(x):
            return np.random.normal(0, 0.4) + x + 4.
        x1 = 4. * np.random.random_sample(DATA_SIZE,) - 4.
        x2 = np.array(map(f, x1))
        t = np.array([[1, 0, 0] for i in xrange(DATA_SIZE)])
        return np.array(zip(x1, x2)), t

    def create_data2():
        def f(x):
            return np.random.normal(0, 0.4) + x
        x1 = 4. * np.random.random_sample(DATA_SIZE,) - 2.
        x2 = np.array(map(f, x1))
        t = np.array([[0, 1, 0] for i in xrange(DATA_SIZE)])
        return np.array(zip(x1, x2)), t

    def create_data3():
        def f(x):
            return np.random.normal(0, 0.4) + x - 4.
        x1 = 4. * np.random.random_sample(DATA_SIZE,)
        x2 = np.array(map(f, x1))
        t = np.array([[0, 0, 1] for i in xrange(DATA_SIZE)])
        return np.array(zip(x1, x2)), t

    X1, T1 = create_data1()
    X2, T2 = create_data2()
    X3, T3 = create_data3()

    W1 = calc_weight(np.r_[X1, X2, X3], np.r_[T1, T2, T3])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))
    ax1.grid(True)
    ax2.grid(True)
    plt.subplots_adjust(wspace = 0.4)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)

    x = np.arange(-10, 10, 0.1)
    x_lower = np.arange(-10, 0, 0.1)
    x_higher = np.arange(0, 10, 0.1)
    border_func1 = get_border(W1[:,:2])
    border1 = np.array(map(border_func1, x))
    ax1.plot(x_lower, map(border_func1, x_lower), 'k')

    border_func2 = get_border(W1[:, 1:])
    border2 = np.array(map(border_func2, x))
    ax1.plot(x_lower, map(border_func2, x_lower), 'k')

    border_func3 = get_border(W1[:, 0::2])
    border3 = np.array(map(border_func3, x))
    ax1.plot(x_higher, map(border_func3, x_higher), 'k')

    ax1.fill_between(x, border1, border2, where=border2>border1, facecolor = 'g', alpha = 0.2)
    ax1.fill_between(x, maximum(border2, border3), 10, facecolor = 'r', alpha = 0.2)
    ax1.fill_between(x, minimum(border1, border3), -10, facecolor = 'b', alpha = 0.2)

    #border_func2 = get_border(W2)
    #ax2.plot(x, map(border_func2, x), 'm')

    ax1.scatter(X1[:,0], X1[:,1], s = 50, c = 'r', marker = "x")
    ax1.scatter(X2[:,0], X2[:,1], s = 50, c = 'g', marker = "x")
    ax1.scatter(X3[:,0], X3[:,1], s = 50, edgecolors = 'b', marker = "o", facecolors= 'none')

    ax2.scatter(X1[:,0], X1[:,1], s = 50, c = 'r', marker = "x")
    ax2.scatter(X2[:,0], X2[:,1], s = 50, c = 'g', marker = "x")
    ax2.scatter(X3[:,0], X3[:,1], s = 50, edgecolors = 'b', marker = "o", facecolors= 'none')

    plt.show()
    #plt.save_fig('fig4_5.png')

if __name__ == '__main__':
    #create_fig4_4()
    create_fig4_5()
