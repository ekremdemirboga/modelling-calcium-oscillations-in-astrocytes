from solver import RHS,rk4,evolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

## Parameters 
vm2 = 20
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

parameters = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
              k_ip3, k_p, k_deg, k_out, k_f, n, m]

# initial conditions
X0 = 0.1
Y0 = 1.5
Z0 = 0.1

initial_conditions = np.array([X0,Y0,Z0])



def solve(parameters, initial_conditions, j):
    ## initial data
    R = initial_conditions

    #simulation parameters
    t_initial = 1
    t_final = 600
    dt = 1/128

    #main part
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)

    #plotting 
    plt.figure()
    plt.plot(time,X,'k',linewidth=3,label = 'X')
    # plt.plot(time,Y,'r',linewidth=3,label = 'Y')
    # plt.plot(time,Z,'g',linewidth=3,label = 'Z')
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    plt.title("parameter is "+ str(parameters[j]))
    # plt.legend(fancybox=True)
    

    
if __name__ == '__main__':
    
    ## parameters
    j = 4 ## j th component of the parameter vector is changed
    # k = 0 ## k th component of the intial condition is changed
    N = 20 ## constant that determines the range of search
    h1 = parameters[j]
    # h2 = initial_conditions[k]
    
    for i in range(N):
        parameters[j] = parameters[j] + h1/N *(i)*(-1)**i
        solve(parameters, initial_conditions, j)
        plt.pause(0.01)
    plt.show()
        