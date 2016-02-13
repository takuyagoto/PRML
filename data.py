# coding:utf-8

from numpy import random

def get_data(size = 1000, min_list = [-1], max_list = [1], params = [-0.3, 0.5], noize_sigma = 0.5, noize_mu = 0):
    data_input = []
    data_output = []

    tmp = []
    for (_min, _max) in zip(min_list, max_list):
        tmp.append(random.uniform(_min, _max, size))
    data_input = map(list, zip(*tmp))

    const = params.pop(0)
    for inpt in data_input:
        ans = const
        for (val, param) in zip(inpt, params):
            ans += val * param
        data_output.append(ans + random.normal(noize_mu, noize_sigma))
    return (data_input, data_output)

if __name__ == '__main__':
    x,t = get_data(size = 10)
    print len(x)
    print x.pop().pop()
    print len(x)

