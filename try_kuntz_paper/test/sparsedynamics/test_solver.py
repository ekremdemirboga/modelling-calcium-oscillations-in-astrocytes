import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc


def RHS(t,R):
    
    X = R[0]
    Y = R[1]
    Z = R[2]
    

    f1 = -4.8*X**2 + 3.5*X*Y  + 3.2*X*Z - 1.6*Y*Z
    f2 = (Y- X)
    f3 = 1

    f = np.array([f1, f2, f3])
    
    return f

def rk4(t,R,RHS,dt):
    ## evolves one time step of RK4 scheme for the vector R(t) -> R(t+ dt)
    k1 = dt*RHS(t,R)
    k2 = dt*RHS(t+dt*0.5, R+k1*0.5)
    k3 = dt*RHS(t+dt*0.5, R+k2*0.5)
    k4 = dt*RHS(t+dt, R+k3)
    return (R + (k1 + 2*k2 + 2*k3 + k4)*(1/6))

def evolve (t_initial, t_final, R, dt):
    
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
        R_next = rk4(t,R,RHS,dt) ## main part
        # save the values and update R
        X_array[i] = R_next[0] 
        Y_array[i] = R_next[1]
        Z_array[i] = R_next[2]
        t_array[i] = t 
        R = R_next
        t +=dt
    return t_array,X_array,Y_array,Z_array

if __name__ == '__main__':
#initial condition
    X0 = 0.01
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
    time,X,Y,Z = evolve(t_initial, t_final, R, dt)

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
    ax.plot3D(X, Y, Z, 'green')


    plt.show()