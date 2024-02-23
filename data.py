import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
mat_file = scipy.io.loadmat('traces.mat')
data_matrix = mat_file["traces"]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def takensEmbedding (data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay*dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')    
    embeddedData = np.array([data[0:len(data)-delay*dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
    return embeddedData


def mutualInformation(data, delay, nBins):
    "This function calculates the mutual information given the delay"
    I = 0
    xmax = max(data)
    xmin = min(data)
    delayData = data[delay:len(data)]
    shortData = data[0:len(data)-delay]
    sizeBin = abs(xmax - xmin) / nBins
    #the use of dictionaries makes the process a bit faster
    probInBin = {}
    conditionBin = {}
    conditionDelayBin = {}
    for h in range(0,nBins):
        if h not in probInBin:
            conditionBin.update({h : (shortData >= (xmin + h*sizeBin)) & (shortData < (xmin + (h+1)*sizeBin))})
            probInBin.update({h : len(shortData[conditionBin[h]]) / len(shortData)})
        for k in range(0,nBins):
            if k not in probInBin:
                conditionBin.update({k : (shortData >= (xmin + k*sizeBin)) & (shortData < (xmin + (k+1)*sizeBin))});
                probInBin.update({k : len(shortData[conditionBin[k]]) / len(shortData)})
            if k not in conditionDelayBin:
                conditionDelayBin.update({k : (delayData >= (xmin + k*sizeBin)) & (delayData < (xmin + (k+1)*sizeBin))});
            Phk = len(shortData[conditionBin[h] & conditionDelayBin[k]]) / len(shortData)
            if Phk != 0 and probInBin[h] != 0 and probInBin[k] != 0:
                I -= Phk * math.log( Phk / (probInBin[h] * probInBin[k]))
    return I


def data_to_embedding(data_matrix, ith_cell,smoothed=False):
    X_data = data_matrix[ith_cell]
    embedded_data = takensEmbedding(X_data,1,3)
    X = embedded_data[0,:]
    Y = embedded_data[1,:]
    Z = embedded_data[2,:]
    if smoothed:
        X = smooth(X,10)
        Y = smooth(Y,10)
        Z = smooth(Z,10)
    return X,Y,Z

if __name__ == '__main__':
    attractor = True
    save = False
    smoothed = True
    mutual_info_test = False
    ith_cell = 82


    if attractor:
        X,Y,Z =data_to_embedding(data_matrix,ith_cell,smoothed)
        if mutual_info_test:
            for i in range(ith_cell):
                X_data = data_matrix[i]
                datDelayInformation = []
                for i in range(1,50):
                    datDelayInformation = np.append(datDelayInformation,[mutualInformation(X_data,i,16)])
                plt.plot(range(1,50),datDelayInformation)
                plt.xlabel('delay')
                plt.ylabel('mutual information')
                plt.show()
        else:
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(X, Y, Z, color = 'red',label = str(ith_cell))
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_zlabel('z', fontsize=12)
            plt.grid(True)
            plt.legend(fancybox=True)
            plt.show()

    else:
        for i in range(1):
            i = ith_cell
            X = data_matrix[i]
            if smoothed:
                X = smooth(X,10)
            plt.figure()
            plt.plot(X,'k',linewidth=3,label = 'X')
            plt.xlabel(r"t", fontsize=14)
            plt.ylabel(r"X", fontsize=14)
            plt.grid(True)
            if save:
                plt.savefig("cell_"+str(i)+"_smoothed.png")
                plt.close()
            else:
                plt.show()  

