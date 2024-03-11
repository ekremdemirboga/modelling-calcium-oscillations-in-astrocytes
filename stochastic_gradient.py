from scipy import stats
from data import data_matrix,data_to_embedding
from grid_search import param_init,param_names,param_init2
from solver import evolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import integrate
import random

R = [0.1,1.5,0.1]
t_initial = 0
t_final = 299
dt = 1/128

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
    smoothed = True
    xx,_,_ =data_to_embedding(data_matrix,cell_id,smoothed)
    # guess = [0.05, 0.05, 0.1, 0.1, 0.3, 0.08, 0.5]
    # result = minimize(objective2,guess,args=(xx,),bounds=bnds)
    # print(result.x)
    
    parameters = param_init2()
    dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.7,(14,))*parameters
    parameters = parameters + dp

    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
    error = np.sum(np.square(X[::130]-xx))
    print(error)
    tol = 7.5
    count = 0
    count2 = 0
    while error>tol:
        time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
        error_new = np.sum(np.square(X[::130]-xx))
        # print(error_new)
        
        if error < error_new:
            parameters = parameters - dp
            dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.7,(14,))*parameters
            parameters = parameters + dp
            count +=1
            if count > 100:
                print("increasing the jump length!")
                dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.01,0.7,(14,))*parameters
        else:
            print(error_new)
            dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.001,0.7,(14,))*parameters
            parameters = parameters + dp
            error = error_new
            count = 0
        if np.isnan(error):
            print("nan occured, increasing jump")
            parameters = parameters - dp
            dp = (2*np.random.randint(0,2,size=(14,))-1)*np.random.uniform(0.01,0.7,(14,))*parameters
            parameters = parameters + dp
        count2 +=1
        if count2>1000:
            break
            
    print("error is: ", error)
    print("parameters are: ", parameters)
    time,X,Y,Z = evolve(t_initial, t_final, R, dt, parameters)
    plt.figure()
    plt.plot(time,X,'k',linewidth=3,label = 'X')
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    plt.show()