from ctypes_loader import calculate_c4y, fitness, gradient
from ctypes import c_int, c_double, byref
import numpy as np
import yaml


class C4weFunction:
    def __init__(self, data, nx):
        self.nx = nx

        count = 0
        for k in range(nx):
            for j in range(k+1):
                for i in range(j+1):
                    if i == 0 and k > 0 and k == j:
                        continue
                    count += 1
        self.num_c4 = count

        n1, n2 = data.shape
        if n1 > n2:
            nchannel, nt = n2, n1
        else:
            nchannel, nt = n1, n2
            data = data.T
        self.nchannel = nchannel
        self.nt = nt

        self.functor_c4y = calculate_c4y()
        self.functor_fitness = fitness()
        self.functor_gradient = gradient()

        self.c4y = np.zeros((self.num_c4, self.nchannel))
        data = np.ascontiguousarray(data)
        self.functor_c4y(data, self.nt, self.nchannel,
                         self.num_c4, self.nx, self.c4y)


    def fitness(self, x):
        return self.functor_fitness(self.c4y, self.num_c4,
                                    self.nchannel, x, self.nx)


    def gradient(self, x):
        grad = np.zeros(self.nx)
        self.functor_gradient(self.c4y, self.num_c4,
                              self.nchannel, x, self.nx, grad)
        return grad
