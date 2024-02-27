import numpy as np
from data import data_matrix,data_to_embedding
from grid_search import param_init
from solver import evolve
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
from scipy.interpolate import interp1d
t_initial = 0
t_final = 299
dt = 1/128
init = np.array([0.1,1.5,0.1]) # initial conditions

def find_dist(cell_id, data_matrix, smoothed = False):
    print("cell #",cell_id)
    ### DATA ###
    xx,yy,zz =data_to_embedding(data_matrix,cell_id,smoothed) ## Taken's embedding of the data

    ### THEORATICAL SOLUTION ###

    parameters = param_init() # initial parameters 
    time,X,Y,Z = evolve(t_initial, t_final, init, dt, parameters)

    curve1= np.array([X[::130],Y[::130],Z[::130]]).T
    curve2= np.array([xx,yy,zz]).T

    distance = average_distance_between_polylines(curve1,curve2)
    print(distance)
    # # Compute pairwise distances between points in the two curves
    # distances = cdist(curve2, curve2)
    
    # # Flatten the distance matrix to obtain the 1D distributions
    # distribution1 = distances.flatten()  # Distribution from curve1
    # distribution2 = distances.T.flatten()  # Distribution from curve2

    # # Compute the Wasserstein distance between the distributions
    # wasserstein_dist = wasserstein_distance(distribution1, distribution2)

    # oneDwasserstein_dist = wasserstein_distance(X, xx)
    # print("Wasserstein distance between the two attractor:", wasserstein_dist)
    return average_distance_between_polylines


def normed_distance_along_path( polyline ):
    polyline = np.asarray(polyline)
    distance = np.cumsum( np.sqrt(np.sum( np.diff(polyline, axis=1)**2, axis=0 )) )
    return np.insert(distance, 0, 0)/distance[-1]

def average_distance_between_polylines(xy1, xy2):   
    s1 = normed_distance_along_path(xy1)
    s2 = normed_distance_along_path(xy2)

    interpol_xy1 = interp1d( s1, xy1 )
    xy1_on_2 = interpol_xy1(s2)

    node_to_node_distance = np.sqrt(np.sum( (xy1_on_2 - xy2)**2, axis=0 ))

    return node_to_node_distance.mean() # or use the max



def objective2(arguments,data_matrix):
    parameters = param_init()
    parameters[0] = arguments[0]
    parameters[1] = arguments[1]
    parameters[2] = arguments[2]
    parameters[3] = arguments[3]
    parameters[4] = arguments[4]
    parameters[5] = arguments[5]
    parameters[6] = arguments[6]
    parameters[7] = arguments[7]
    parameters[8] = arguments[8]
    parameters[9] = arguments[9]
    parameters[10] = arguments[10]
    parameters[11] = arguments[11]
    parameters[12] = arguments[12]
    parameters[13] = arguments[13]
    xx,yy,zz =data_to_embedding(data_matrix,81,False) ## Taken's embedding of the data
    time,X,Y,Z = evolve(t_initial, t_final, init, dt, parameters)
    curve1= np.array([X[::130],Y[::130],Z[::130]]).T
    curve2= np.array([xx,yy,zz]).T
    
    error = average_distance_between_polylines(curve1,curve2)
    return error


bnds = [(0, None), (0, None), (0, 1), (0,1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1), (0,1), (0, 1), (0, 1), (0, 1)]

guess = param_init()
result = minimize(objective2,guess,args=(data_matrix,),bounds=bnds)
print(result.x)

