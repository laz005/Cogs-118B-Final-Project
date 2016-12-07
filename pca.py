import numpy as np
import numpy.matlib as npmatlib
import pickle
import matplotlib.pyplot as plt

#Load data -- TEMPORARY
data = pickle.load(open("ClevelandProcData.p","rb"))
N = len(data)

#Turn all decimal data into float data
for a in range(0, N):
    for b in range(0,len(data[a])):
        data[a][b] = float(data[a][b])
#Seperate data from label for crossvalidation
dataLabels = [row[13] for row in data]
for dat in range(0, N):
    del data[dat][13]

def runPCA(data):
    #print data

    mean = np.mean(data,axis=0)
    means = npmatlib.repmat(mean, len(data), 1)

    A = data - means            # 303x13
    Atrans = np.transpose(A)    # 13x303

    eigvals, eigvecs = np.linalg.eig( np.mat(Atrans)*np.mat(A) )

    #print w
    #print v

    # sort eigenvectors
    index = np.argsort(eigvals)
    indexSorted = index[::-1]

    # top 2 eigenvalues and eigenvectors
    #print "eigenvalues:"
    #print eigvals[indexSorted[0:2]] # 1x2 matrix
    #print "eigenvectors:"
    #print eigvecs[0:,indexSorted[0:2]] # 13x2 matrix
    eigvecsTrans = np.transpose(eigvecs[0:,indexSorted[0:2]]) # 2x13
    # transformed data
    C = np.mat(eigvecsTrans)* np.mat(Atrans) # 2x13 * 13x303 = 2x303

    # convert to array
    x = np.asarray(C[0,0:]).reshape(-1)
    y = np.asarray(C[1,0:]).reshape(-1)
    colors = ['co', 'bo', 'ro', 'go', 'mo']

    for i in xrange(1, len(dataLabels)):
        plt.plot(x[i], y[i], colors[int(dataLabels[i])])

    plt.axis([-150,160,-80,90])
    plt.show()

runPCA(data)
