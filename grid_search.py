from solver import RHS,rk4,evolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
from matplotlib import colormaps

number_of_parameters=14

# initial conditions
X0 = 0.1
Y0 = 1.5
Z0 = 0.1

initial_conditions = np.array([X0,Y0,Z0])

def solve(parameters, initial_conditions, param_code , color_code, N,ax, attractor=False,):
    ## initial data
    R = initial_conditions

    #simulation parameters
    t_initial = 1
    t_final = 600
    dt = 1/128

    #main part
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)

    #plotting 
    cmap = plt.get_cmap('OrRd')
    rgba = cmap(np.linspace(0.3,1,N))
    
    if attractor == True:
        ax.plot3D(X, Y, Z, color = rgba[color_code],label = str(round(parameters[param_code], 2)))
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
    else:        
        plt.plot(time,X,color = rgba[color_code] ,linewidth=2,label = str(round(parameters[param_code], 2)))
        plt.xlabel(r"t", fontsize=14)
        plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    
def param_names(j):
    # return the name of the jth parameter
    # [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
    #           k_ip3, k_p, k_deg, k_out, k_f, n, m]
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

def param_init():
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
    n = 2.0
    m = 2.2
    params = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
              k_ip3, k_p, k_deg, k_out, k_f, n, m]
    return np.asarray(params)

def grid_search(param_code,N,saveflag=True):
    ## Makes a grid search for the parameter given by param_names(param_code)
    # limits of search of parameters
    parameters = param_init()
    step = (parameters[param_code]/N)
    start = parameters[param_code] - N*(step)/2
    end = parameters[param_code] + N*(step)/2

    Attractor = True
    save_loc = "plots/"

    plt.figure()
    if Attractor:
        ax = plt.axes(projection='3d')
    else:
        ax = 0
    for i in range(N):
        parameters[param_code] = start + step*i
        solve(parameters, initial_conditions, param_code, i, N,ax,Attractor)
    plt.title("parameter: "+param_names(param_code))
    plt.legend(fancybox=True)
    if saveflag:
        if Attractor:
            plt.savefig(save_loc+"parameter_attractor-"+param_names(param_code)+".png")
        else:
            plt.savefig(save_loc+"parameter-"+param_names(param_code)+".png")
        parameters = param_init()  ## reseting the parameters
    else:
        plt.show()    
if __name__ == '__main__':
    ## parameters
    #param_code = 1 ## th component of the parameter vector is changed
    N = 4 ## constant that determines the range of search
    save_flag = True
    
    for param_code in range(number_of_parameters):
        print("looking for parameter "+param_names(param_code))
        grid_search(param_code,N,save_flag)
    