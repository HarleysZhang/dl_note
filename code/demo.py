# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def target_function(x,y):
    J = pow(x, 2) + 2*pow(y, 2)
    return J

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x, 4*pow(y, 1)])

def show_3d_surface(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    u = np.linspace(-3, 3, 100)
    v = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(u, v)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = pow(X[i, j], 2)+ 4*pow(Y[i, j], 2)

    ax.plot_surface(X, Y, R, cmap='rainbow')
    plt.plot(x, y, z, c='black', linewidth=1.5,  marker='o', linestyle='solid')
    plt.show()

if __name__ == '__main__':
    theta = np.array([-3, -3]) # 输入为双变量
    eta = 0.1 # 学习率
    error = 5e-3 # 迭代终止条件，目标函数值 < error

    X = []
    Y = []
    Z = []
    for i in range(50):
        print(theta)
        x = theta[0]
        y = theta[1]
        z = target_function(x,y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("%d: x=%f, y=%f, z=%f" % (i,x,y,z))
        d_theta = derivative_function(theta)
        print("    ", d_theta)
        theta = theta - eta * d_theta
        if z < error:
            break
    show_3d_surface(X,Y,Z)