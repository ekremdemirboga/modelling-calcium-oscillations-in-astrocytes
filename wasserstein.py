from scipy import stats
from data import data_matrix,data_to_embedding
from grid_search import param_init,param_names
from solver import evolve
import numpy as np
import matplotlib.pyplot as plt
from cmaes import CMA
from scipy.optimize import minimize

### DATA ###
smoothed = False ### flag for smoothing of the data
cell_id = 82 
xx,yy,zz =data_to_embedding(data_matrix,cell_id,smoothed) ## Taken's embedding of the data

### THEORATICAL SOLUTION ###

# initial conditions
X0 = 0.1
Y0 = 1.5
Z0 = 0.1
t_initial = 1
t_final = 300
dt = 1/128
R = np.array([X0,Y0,Z0]) # initial conditions
parameters = param_init() # initial parameters 
## params = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m]
N = 20 ## constant that determines the range of search
number_of_parameters=14
dp = parameters/2

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



def will_be_optimized(arguments):
    
    parameters = param_init()
    parameters[0] = arguments[0]
    parameters[1] = arguments[1]
    parameters[6] = arguments[2]
    parameters[5] = arguments[3]
    # parameters[8] = arguments[2]
    cell_id = 81
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
    xx,yy,zz =data_to_embedding(data_matrix,cell_id,smoothed)
    a = stats.wasserstein_distance(X,xx)
    return a


if __name__ == "__main__":
    bnds = ((0, None), (0, None), (0, 5), (0, 5))
    # initial = param_init()
    #parameters = np.array([vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m])
    result = minimize(will_be_optimized,[15,40,0.15,0.15], bounds=bnds)
    print(result.x)
    # res = minimize(fun,[0.15,0.15,0,3], method='SLSQP', bounds=bnds)
