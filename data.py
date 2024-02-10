import scipy.io
import matplotlib.pyplot as plt

mat_file = scipy.io.loadmat('traces.mat')
data_matrix = mat_file["traces"]

X = data_matrix[5]

plt.figure()
plt.plot(X,'k',linewidth=3,label = 'X')
plt.xlabel(r"t", fontsize=14)
plt.ylabel(r"X", fontsize=14)
plt.grid(True)
plt.show()