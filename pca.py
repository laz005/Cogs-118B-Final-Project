import numpy as np

def runPCA(data):
    mean = np.mean(data,axis=0)
    means = np.matlib.repmat(mean, len(data), 1)

    A = data - means
    Atrans = np.transpose(A)


    w,v = np.linalg.eig( np.mat(Atrans)*np.mat(A) )

    print w
    print v