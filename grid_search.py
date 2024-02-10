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
    
    plt.plot(time,X,linewidth=3,label = str(round(parameters[j], 2)))
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    # plt.title("parameter is "+ str(parameters[j]))
    
def param_names(j):
    [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
              k_ip3, k_p, k_deg, k_out, k_f, n, m]
    if j==0:
        name = "vm2"
    elif j==1:
        name = "vm3"
    elif j==2:
        name = "v_in"
    elif j==3:
        name = "v_p"
    elif j==4:
        name = "k_2"
    elif j==5:
        name = "k_CaA"
    elif j==6:
        name = "k_CaI"
    elif j==7:
        name = "k_ip3"
    elif j==8:
        name = "k_p"
    elif j==9:
        name = "k_deg"
    elif j==10:
        name = "k_out"
    elif j==11:
        name = "k_f"
    elif j==12:
        name = "n"
    elif j==13:
        name = "m"
    else:
        print("invalid j")
    return name
        
        
    
if __name__ == '__main__':
    ## parameters
    j = 1 ## j th component of the parameter vector is changed
    N = 6 ## constant that determines the range of search
    step = (parameters[j]/N)
    start = parameters[j] - N*(step)/2
    end = parameters[j] + N*(step)/2
    
    # k = 0 ## k th component of the intial condition is changed
    # h2 = initial_conditions[k]
    plt.figure()
    for i in range(N):
        parameters[j] = start + step*i
        solve(parameters, initial_conditions, j)
        # plt.pause(0.001)
    plt.title("parameter: "+param_names(j))
    plt.legend(fancybox=True)
    plt.show()
        