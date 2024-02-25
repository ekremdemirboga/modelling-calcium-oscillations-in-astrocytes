import numpy as np
from data import data_matrix,data_to_embedding
from grid_search import param_init
from solver import evolve
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


### DATA ###
smoothed = False ### flag for smoothing of the data
cell_id = 82 
xx,yy,zz =data_to_embedding(data_matrix,cell_id,smoothed) ## Taken's embedding of the data
### THEORATICAL SOLUTION ###
t_initial = 0
t_final = 300
dt = 1/128
init = np.array([0.1,1.5,0.1]) # initial conditions
parameters = param_init() # initial parameters 
time,X,Y,Z = evolve(t_initial, t_final, init, dt, parameters)

curve1= np.array([X,Y,Z]).T
curve2= np.array([xx,yy,zz]).T

print(curve1.shape)
print(curve2.shape)



# Compute pairwise distances between points in the two curves
distances = cdist(curve2, curve2)
print(distances)
# Flatten the distance matrix to obtain the 1D distributions
distribution1 = distances.flatten()  # Distribution from curve1
distribution2 = distances.T.flatten()  # Distribution from curve2

# Compute the Wasserstein distance between the distributions
wasserstein_dist = wasserstein_distance(distribution1, distribution2)

print("Wasserstein distance between the two attractor:", wasserstein_dist)

distance_X = wasserstein_distance(X,xx)
print("1D Wasserstein distance between the two X curves:", distance_X)