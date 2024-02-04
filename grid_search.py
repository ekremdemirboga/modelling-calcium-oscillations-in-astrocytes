from solver import RHS,rk4,evolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

## Parameters 
vm2 = 15
vm3 = 40
v_in = 0.05
v_p = 0.05
k_2 = 0.1
k_CaA = 0.15
k_CaI = 0.15
k_ip3 = 0.1
k_p = 0.3
k_deg = 0.08
k_out = 0.5
k_f = 0.5
n = 2
m = 2

# initial conditions
X0 = 0.1
Y0 = 1.5
Z0 = 0.1

parameters = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
              k_ip3, k_p, k_deg, k_out, k_f, n, m]

initial_conditions = np.array([X0,Y0,Z0])

def plot(parameters, initial_conditions):
    ## initial data
    R = initial_conditions

    #simulation parameters
    t_initial = 1
    t_final = 600
    dt = 1/64

    #main part
    time,X,Y,Z = evolve(t_initial, t_final, R, dt)

    #plotting orbit
    plt.figure()
    plt.plot(time,X,'k',linewidth=3,label = 'X')
    # plt.plot(time,Y,'r',linewidth=3,label = 'Y')
    # plt.plot(time,Z,'g',linewidth=3,label = 'Z')
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    # plt.legend(fancybox=True)


    plt.show()