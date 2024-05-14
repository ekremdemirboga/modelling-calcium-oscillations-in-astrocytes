import numpy as np
import matplotlib.pyplot as plt


def RHS(t,R,parameters,noise = False):
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
    noise2 = 0
    if noise:
        gaussian_noise = np.random.normal(0, 0.4)
        v_in=v_in+gaussian_noise
        noise2 =  np.random.normal(0, 4)
    f1 = v_in- k_out*X + v_CICR - v_serca + k_f * (Y - X)
    f2 = v_serca - v_CICR - k_f * (Y- X) +noise2
    f3 = v_PLC - k_deg * Z

    f = np.array([f1, f2, f3])
    
    return f

def rk4(t,R,RHS,dt,parameters,noise=False):
    ## evolves one time step of RK4 scheme for the vector R(t) -> R(t+ dt)
    k1 = dt*RHS(t,R, parameters,noise)
    k2 = dt*RHS(t+dt*0.5, R+k1*0.5, parameters,noise)
    k3 = dt*RHS(t+dt*0.5, R+k2*0.5, parameters,noise)
    k4 = dt*RHS(t+dt, R+k3, parameters,noise)
    return (R + (k1 + 2*k2 + 2*k3 + k4)*(1/6))


def evolve (t_initial, t_final, R, dt, parameters,noise=False):
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
        if i%50 == 0 and noise:
            n = True
        else:
            n = False         
        R_next = rk4(t,R,RHS,dt,parameters,n) ## main part
        # save the values and update R
        X_array[i] = R_next[0] 
        Y_array[i] = R_next[1]
        Z_array[i] = R_next[2]
        # t_array[i] = t 
        R = R_next
        t +=dt
    return t_array,X_array,Y_array,Z_array


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
    n = 2.02
    m = 2.2
    params = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
              k_ip3, k_p, k_deg, k_out, k_f, n, m]
    return np.asarray(params)

if __name__ == '__main__':
    ##Reproducing the results from https://www.sciencedirect.com/science/article/pii/S0022519307006510?via%3Dihub
    
    # vm2 = 15
    # vm3 = 40
    # v_in = 0.05
    # v_p = 0.05
    # k_2 = 0.1
    # k_CaA = 0.27
    # k_CaI = 0.27
    # k_ip3 = 0.1
    # k_p = 0.164
    # k_deg = 0.08
    # k_out = 0.5
    # k_f = 0.5
    # n = 2.02
    # m = 2.2
    
    vm2 = 14.73473573752892 # fixed
    vm3 = 42.27124247928102 # fixed
    v_in =  0.010000000006538988
    v_p = 0.011020778109060726
    k_2 = 0.04833079640985642
    k_CaA = 0.33137470492398174 #fixed
    k_CaI = 0.05554903743145177 #fixed
    k_ip3 = 0.07052629389235288
    k_p = 0.31123713910093365
    k_deg = 0.010000161695752763
    k_out = 0.5303514867643699 ## fixed
    k_f = 0.36990349669220096 ##11
    n = 2.1787746328706192 #fixed
    m = 2.158006702188488  #fixed 
    # parameters = np.array([vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m])
    parameters = param_init()
    #initial condition
    X0 = 0.1
    Y0 = 4.9
    Z0 = 0.01
    initial_conditions = np.array([X0,Y0,Z0])
    ## initial data
    R = initial_conditions

    #parameters
    t_initial = 0
    t_final = 300
    dt = 1/128
    noise = False
    #main part
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters,noise)

    #plotting 
    plt.figure()
    plt.plot(time,X,'k',linewidth=3,label = 'X')
    # plt.plot(time,Y,'r',linewidth=3,label = 'Y')
    # plt.plot(time,Z,'g',linewidth=3,label = 'Z')
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    # plt.legend(fancybox=True)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X, Y, Z, 'black')


    plt.show()