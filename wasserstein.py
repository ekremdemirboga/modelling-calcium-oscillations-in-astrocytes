from scipy import stats
from data import data_matrix,data_to_embedding
from grid_search import param_init,param_names
from solver import evolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import integrate
import random

### DATA ###
smoothed = True ### flag for smoothing of the data
cell_id = 82 
xx,yy,zz =data_to_embedding(data_matrix,cell_id,smoothed) ## Taken's embedding of the data

### THEORATICAL SOLUTION ###

# initial conditions
X0 = 0.1
Y0 = 1.5
Z0 = 0.1
t_initial = 0
t_final = 299
dt = 1/128
R = np.array([X0,Y0,Z0]) # initial conditions
parameters = param_init() # initial parameters 
## params = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m]
# N = 20 ## constant that determines the range of search
# number_of_parameters=14
# dp = parameters/2

# time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
# distance_old = stats.wasserstein_distance(X,xx)
# print(distance_old)
# plt.figure()
# plt.plot(time,X)
# plt.plot(xx)
# plt.show()


# parameters = param_init()
# t1,a,_,_ = evolve(t_initial, t_final, R, dt, parameters)
# parameters = parameters + 1e-3
# t2,b,_,_ = evolve(t_initial, t_final, R, dt, parameters)
# print(stats.wasserstein_distance(a[::2],b))

# plt.figure()
# plt.plot(t1[::2],a[::2],t2,b)
# plt.show()


# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(xx, yy, zz, color = 'red',label = "data "+str(cell_id))
# ax.plot3D(X[::130],Y[::130],Z[::130], color = 'green',label = "solution")
# ax.set_xlabel('x', fontsize=12)
# ax.set_ylabel('y', fontsize=12)
# ax.set_zlabel('z', fontsize=12)
# plt.grid(True)
# plt.legend(fancybox=True)
# plt.show()



def objective(arguments,data):
    
    parameters = param_init()
    parameters[0] = arguments[0]
    parameters[1] = arguments[1]
    parameters[5] = arguments[2]
    parameters[6] = arguments[3]
    
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
    
    error = stats.wasserstein_distance(X,data)
    print(error)
    return error

def objective2(arguments,data):
    parameters = param_init()
    # parameters[0] = arguments[0]
    # parameters[1] = arguments[1]
    parameters[2] = arguments[0]
    parameters[3] = arguments[1]
    parameters[4] = arguments[2]
    # parameters[5] = arguments[5]
    # parameters[6] = arguments[6]
    parameters[7] = arguments[3]
    parameters[8] = arguments[4]
    parameters[9] = arguments[5]
    # parameters[10] = arguments[10]
    parameters[11] = arguments[6]
    # parameters[12] = arguments[12]
    # parameters[13] = arguments[13]
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)

    error = np.sum(np.square(X[::130]-data))
    return error



if __name__ == "__main__":
    bnds = [(0, 1), (0, 1), (0, 1), (0,1), (0, 1), (0, 1), (0, 1)]
    cell_id = 81
    xx,_,_ =data_to_embedding(data_matrix,cell_id,smoothed)
    # guess = [0.05, 0.05, 0.1, 0.1, 0.3, 0.08, 0.5]
    # result = minimize(objective2,guess,args=(xx,),bounds=bnds)
    # print(result.x)
    noise = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.2,(14,))
    parameters = param_init()
    dp = noise*parameters
    parameters = parameters + dp
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
    error = np.sum(np.square(X[::130]-xx))
    print(error)
    tol = 1e-1
    count = 0
    while error>tol:
        time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
        error_new = np.sum(np.square(X[::130]-xx))
        # print(error_new)
        
        if error < error_new:
            parameters = parameters - dp
            dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.2,(14,))*parameters
            parameters = parameters + dp
            count +=1
            if count > 100:
                dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.1,0.5,(14,))*parameters
        else:
            print(error_new)
            dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.2,(14,))*parameters
            parameters = parameters + dp
            error = error_new
            count = 0
        if np.isnan(error):
            parameters = parameters - dp
            dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.5,(14,))*parameters
            parameters = parameters + dp
            error = 999
            
    print("error is: ", error)
    print("parameters are: ", parameters)
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
    plt.figure()
    plt.plot(time,X,'k',linewidth=3,label = 'X')
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)



    plt.show()