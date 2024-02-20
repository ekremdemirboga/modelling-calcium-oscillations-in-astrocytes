import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc


def RHS(t,R,parameters):
    ## caclulates the RHS of the equation 28 in the https://www.sciencedirect.com/science/article/pii/S0022519307006510?via%3Dihub.
    # INPUTS 
    ## parameters: constants of the ODE
    ## t: time  
    ## R = (X, Y, Z) where
    # OUTPUTS 
    ## f = (f1, f2, f3) where
    ## f#'s are the RHS of the equation
    
    [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m] = parameters
    
    X = R[0]
    Y = R[1]
    Z = R[2]
    
    v_serca = (vm2 * X**2) / (X**2 + k_2 **2)    
    v_PLC = (v_p * X**2) / (X**2 + k_p **2) 
    v_CICR = 4*vm3*(k_CaA**n * X**n / ( (X**n + k_CaA**n)*(X**n + k_CaI**n) ) )*(Z**m/(Z**m + k_ip3**m))*(Y-X)
    
    f1 = v_in - k_out*X + v_CICR - v_serca + k_f * (Y - X)
    f2 = v_serca - v_CICR - k_f * (Y- X)
    f3 = v_PLC - k_deg * Z 

    f = np.array([f1, f2, f3])
    
    return f

def rk4(t,R,RHS,dt,parameters):
    ## evolves one time step of RK4 scheme for the vector R(t) -> R(t+ dt)
    k1 = dt*RHS(t,R, parameters)
    k2 = dt*RHS(t+dt*0.5, R+k1*0.5, parameters)
    k3 = dt*RHS(t+dt*0.5, R+k2*0.5, parameters)
    k4 = dt*RHS(t+dt, R+k3, parameters)
    return (R + (k1 + 2*k2 + 2*k3 + k4)*(1/6))


def evolve (t_initial, t_final, R, dt, parameters):
    ## evolves the vector R from the initial time to the final time, R(t_initial) -> R(t_final)
    ## using rk4.

    t = t_initial ## starting time

    ## define arrays for storing the solution
    t_array = np.arange(t_initial, t_final+dt, dt)
    X_array = np.zeros((len(t_array)),dtype=np.float64)
    Y_array = np.zeros((len(t_array)),dtype=np.float64)
    Z_array = np.zeros((len(t_array)),dtype=np.float64)
    # Set initial values
    X_array[0] = R[0]
    Y_array[0] = R[1]
    Z_array[0] = R[2]
    
    ## evolve by iterating rk4 len(t_array)
    for i in range(1, len(t_array)):        
        R_next = rk4(t,R,RHS,dt,parameters) ## main part
        # save the values and update R
        X_array[i] = R_next[0] 
        Y_array[i] = R_next[1]
        Z_array[i] = R_next[2]
        t_array[i] = t 
        R = R_next
        t +=dt
    return t_array,X_array,Y_array,Z_array



if __name__ == '__main__':
    ##Reproducing the results from https://www.sciencedirect.com/science/article/pii/S0022519307006510?via%3Dihub
    
    ## Constants 
    vm2 = 20
    vm3 = 40
    v_in = 0.055
    v_p = 0.05
    k_2 = 0.1
    k_CaA = 0.27
    k_CaI = 0.27
    k_ip3 = 0.1
    k_p = 0.164
    k_deg = 0.08
    k_out = 0.5
    k_f = 0.5
    n = 2.0
    m = 2.2
    parameters = np.array([vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m])
    
    #initial condition
    X0 = 0.1
    Y0 = 1.5
    Z0 = 0.1
    initial_conditions = np.array([X0,Y0,Z0])
    ## initial data
    R = initial_conditions

    #parameters
    t_initial = 1
    t_final = 300
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
    # plt.legend(fancybox=True)


    plt.show()