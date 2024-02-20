import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

mat_file = scipy.io.loadmat('traces.mat')
data_matrix = mat_file["traces"]
for i in range(len(data_matrix)):
    X = data_matrix[i]
    plt.figure()
    X = smooth(X,10)
    plt.plot(X,'k',linewidth=3,label = 'X')
    plt.xlabel(r"t", fontsize=14)
    plt.ylabel(r"X", fontsize=14)
    plt.grid(True)
    plt.savefig("cell_"+str(i)+"_smoothed.png")
    plt.close()
    # plt.show()