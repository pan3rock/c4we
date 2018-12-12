#!/usr/bin/env python
from c4we_fn import C4weFunction
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from obspy import read
import math
from ctypes_loader import calculate_c4y
from scipy.signal import butter, filtfilt


def initialize_random(nx):
    ret = np.zeros(nx)
    sdev = 0.1
    for i in range(0, nx, 2):
        value = np.random.random()
        rho = sdev * math.sqrt(2.0 * abs(math.log(value)))
        theta = 2.0 * math.pi * np.random.random()
        ret[i] = rho * math.cos(theta)
        ret[i+1] = rho * math.sin(theta)
    ret[nx // 2] = 1.0
    return ret


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inversion")
    parser.add_argument("--maxiter",
                        type=int,
                        default=100,
                        help="maxmium number of iterations[100]")
    parser.add_argument("--nx",
                        type=int,
                        help="number of wavelet samplings")
    parser.add_argument("--ftol",
                        type=float,
                        default=1.0e-5,
                        help="The iteration stops when (f^k - f^{k+1})"
                        +"/max{|f^k|,|f^{k+1}|,1} <= ftol.")
    args = parser.parse_args()
    maxiter = args.maxiter
    ftol = args.ftol
    nx = args.nx

    st = read("./M2.sg2")
    data_list = []
    nr = st.count()
    for i in range(nr):
        data_list.append(st.traces[i].data)
    data = np.array(data_list, dtype=np.float64)
    time = st.traces[0].times()
    nt = time.shape[0]
    dt = time[1] - time[0]
    # data = np.load("./data_l4_wang2.npz")
    # time = data['time']
    # data = data['vz']
    # nr = data.shape[0]
    # nt = time.shape[0]
    # dt = time[1] - time[0]

    # for i in range(nr):
    #     data[i, :] = butter_lowpass_filter(data[i, :], 60., 1./dt)

    # data = data[0, :].reshape((1, -1))

    x_final_list = []
    x0 = initialize_random(nx)
    print(data.shape)
    for i in range(nr):
        trace = data[i, :].reshape((1, -1))
        obj = C4weFunction(trace, nx)
        method = "L-BFGS-B"
        res = minimize(obj.fitness, x0, jac=obj.gradient,
                       method=method,
                       options={'disp': True,
                                'ftol': ftol,
                                'maxiter': maxiter})
        x_final_list.append(res.x)
    x_final = np.mean(np.array(x_final_list), axis=0)

    m = nx
    y = x_final
    nt_1 = m - 1
    mlag = (nt_1**3 + 6 * nt_1**2 + 11 * nt_1 + 6) // 6
    mf4 = np.zeros(mlag)
    count = 0
    for k in range(m):
        for j in range(k+1):
            for i in range(j+1):
                if i == 0 and k > 0 and k == j:
                    continue
                vsum = 0
                for n in range(m - k):
                    vsum += y[n] * y[n + i] * y[n + j] * y[n + k]
                mf4[count] = vsum
                count += 1

    c4y = np.zeros((mlag, nr))
    func = calculate_c4y()
    func(data, nt, nr, mlag, nx, c4y)
    plt.figure()
    plt.plot(mf4, 'r', alpha=0.5)
    plt.plot(c4y.flatten(), 'b', alpha=0.5)
    plt.show()

    # plotting
    time = np.arange(nx) * dt
    print(res.x)
    plt.figure()
    plt.plot(time, x0, 'r', label="x0", alpha=0.8)
    for i in range(nr):
        plt.plot(time, x_final_list[i], 'k', alpha=0.5)
    plt.plot(time, x_final, 'b', label="x_final", alpha=0.8)
    plt.legend()
    plt.show()
