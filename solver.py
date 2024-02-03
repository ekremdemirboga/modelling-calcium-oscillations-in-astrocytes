import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

## Constants 
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
X0 = 0.1
Y0 = 1.5
Z0 = 1.5

def RHS(t,R):
    ## caclulates the RHS of the equation 28 in the https://www.sciencedirect.com/science/article/pii/S0022519307006510?via%3Dihub.
    # INPUTS 
    ## t: time  
    ## R = (X, Y, Z) where
    # OUTPUTS 
    ## f = (f1, f2, f3) where
    ## f#'s are the RHS of the equation 

    X = R[0]
    Y = R[0]
    Z = R[0]
    v_serca = vm2 * X**2 / (X**2 + k_2 **2)    
    v_PLC = v_p * X**2 / (X**2 + k_2 **2) 
    v_CICR = 4*vm3 * k_CaA**n * X**n * Z**m * (Y-X) / ( (X**n + k_CaA**n ) * (X**n + k_CaI**n) * (Z**m *k_ip3**m))

    def f1(t,r,rdot):
        return rdot
    
    def f2(t,r,rdot):
        rhat = r/np.linalg.norm(r)
        return -(G * m)/(np.linalg.norm(r)**2) * rhat - (1/c**5) *  ( (6.0 * G**3 * m**3)/(5.0*np.linalg.norm(r)**4) 
                                                                 + np.linalg.norm(rdot)**2*(2.0 * G**2 * m**2 ) / (5.0 * np.linalg.norm(r)**3)   )*rdot
    f = np.array([f1(t,R[0],R[1]) , f2(t,R[0],R[1])])
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
    r_array = np.zeros((len(t_array),2),dtype=np.float64)
    v_array = np.zeros((len(t_array),2),dtype=np.float64)
    # Set initial values
    r_array[0] = R[0]
    v_array[0] = R[1]

    ## flag for determining the point where the distance is 5m
    flag = 1
    x_at5 = 0
    y_at5 = 0
    T5 = 0
    ## evolve by iterating rk4
    for i in range(1, len(t_array)):
        R_next = rk4(t,R,RHS,dt) ## main part
        if np.linalg.norm(R_next[0])<=5.0*m and flag: ## check whether distance d<5m.
            print("seperation is "+ str(np.linalg.norm(R_next[0])))
            print("time elapsed: " +str(t))
            ## position and time information when the distance d=5m
            x_at5=R_next[0,0]
            y_at5=R_next[0,1]
            T5 = t
            flag = 0
        # save the values and update R
        r_array[i] = R_next[0] 
        v_array[i] = R_next[1]
        t_array[i] = t 
        R = R_next
        t +=dt
    return t_array,r_array,v_array,x_at5,y_at5,T5


if __name__ == '__main__':
    ## initial data
    r0 = np.array([0,10.0*m])
    v0 = np.array([1/np.sqrt(10),0])
    R = np.array([r0,v0])

    #parameters
    t_initial = 0
    t_final = 800
    dt = 1/128

    #main part
    time,r,velocity,x_at5,y_at5,T5 = evolve(t_initial, t_final, R, dt)

    #plotting orbit
    plt.figure()
    plt.plot(r[:,0],r[:,1],'k',linewidth=3)
    plt.scatter(x_at5,y_at5,label=r'$|\vec{r}|=5m$',color='red')
    plt.plot([0,x_at5],[0,y_at5],'r--',linewidth=2)
    plt.text(2.5, .5, r'$|\vec{r}|=5m$',rotation = 4,
         rotation_mode = 'anchor',horizontalalignment='center', fontsize=12)
    plt.xlabel(r"x", fontsize=14)
    plt.ylabel(r"y", fontsize=14)
    plt.grid(True)
    plt.legend(fancybox=True)

    #plotting the distance as a function of time
    plt.figure()
    plt.plot(time,np.sqrt(r[:,0]**2 +r[:,1]**2),'r',linewidth=3,label='distance')
    plt.xlabel(r"time (in m)",fontsize=14)
    plt.ylabel(r"$|\vec(r)|$ (in m)",fontsize=14)
    plt.axvline(x = T5, color = 'k',linestyle='--', label = r'T5='+str(T5))
    plt.grid(True)
    plt.legend(fancybox=True)

    plt.show()