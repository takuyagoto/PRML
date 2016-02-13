#!/usr/bin/python
#coding:utf-8

import numpy as np
from matplotlib import pyplot as plt


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

"""
最小二乗の境界
"""
def get_border(W):
    def func(x):
        return (-x * (W[1][0] - W[1][1]) - (W[0][0] - W[0][1]))/(W[2][0] - W[2][1])
    return func


"""
Fisher's discriminant
"""
def calc_mean(x, dimension):
    m = np.zeros(dimension)
    for xn in x:
        m += xn
    return m / len(x)

def calc_S_total(x, mean):
    S_T = np.zeros((len(mean), len(mean)))
    for xn in x:
        diff = xn - mean
        S_T += diff[:, np.newaxis] * diff
    return S_T

def calc_S_between(mean_total, means, size_list):
        dimension = len(mean_total)
        S_B = np.zeros((dimension, dimension))
        for mean, size in zip(means, size_list):
            diff = mean - mean_total
            S_B += size * diff[:, np.newaxis] * diff
        return S_B

def calc_Fisher_weight(x_list, y_list, dimension, k = 2):
    size_list = [len(x) for x in x_list]
    means = [calc_mean(x, dimension) for x in x_list]
    mean_total = np.zeros(dimension)
    for mean, size in zip(means, size_list):
        mean_total += mean * size
    mean_total /= sum(size_list)
    x_all = []
    for x in x_list:
        x_all += x
    S_T = calc_S_total(x_all, mean_total)
    S_B = calc_S_between(mean_total, means, size_list)
    S_W = S_T - S_B
    eig_val, eig_vec = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    sorted_eig_val_index = np.argsort(eig_val)
    sorted_eig_val_index = sorted_eig_val_index[-1::-1]
    weight = []
    for index in sorted_eig_val_index[:(k-1)]:
        weight.append(eig_vec[:, index])
    return np.array(weight), mean_total

def get_Fisher_border(weight, mean):
    def f(x):
        return (- weight[0] / weight[1]) * (x - mean[0]) + mean[1]
    return f

def create_fig_outlier():
    def create_data1():
        def f(x):
            return np.random.normal(0, 0.4) + x + 2.
        x1 = 6. * np.random.random_sample(DATA_SIZE,) - 4.
        x2 = np.array(map(f, x1))
        t = np.array([[1, 0] for i in xrange(DATA_SIZE)])
        return zip(x1, x2), t

    def create_data2():
        def f(x):
            return np.random.normal(0, 0.4) + x - 2.
        x1 = 6. * np.random.random_sample(DATA_SIZE,) - 2.
        x2 = np.array(map(f, x1))
        t = np.array([[0, 1] for i in xrange(DATA_SIZE)])
        return zip(x1, x2), t

    def create_outlier():
        x1 = np.array([np.random.normal(8., 0.8) for i in xrange(OUTLIER_SIZE)])
        x2 = np.array([np.random.normal(-6., 0.8) for i in xrange(OUTLIER_SIZE)])
        t = np.array([[0, 1] for i in xrange(OUTLIER_SIZE)])
        return zip(x1, x2), t

    X1, T1 = create_data1()
    X2, T2 = create_data2()
    X3, T3 = create_outlier()

    W_f1, mean1 = calc_Fisher_weight([X1,X2], [T1, T2], 2, 2)
    W_f2, mean2 = calc_Fisher_weight([X1,X2, X3], [T1, T2, T3], 2, 2)

    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
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
    Fisher_border_func1 = get_Fisher_border(W_f1[0], mean1)
    ax1.plot(x, map(border_func1, x), 'm')
    ax1.plot(x, map(Fisher_border_func1, x), 'k')

    border_func2 = get_border(W2)
    Fisher_border_func2 = get_Fisher_border(W_f2[0], mean2)
    ax2.plot(x, map(border_func2, x), 'm')
    ax2.plot(x, map(Fisher_border_func2, x), 'k')

    plt.show()
    plt.savefig('fig_fisher.png')


if __name__ == '__main__':
    create_fig_outlier()
    # create_fig_multi_class()
