import numpy.matlib
import pickle
from random import sample as randsample
import pca

numpy.set_printoptions(precision=2)
numpy.set_printoptions(suppress=True)

def getoccurence(s,h):
    return [i for i, letter in enumerate(s) if letter == h]
# Manhattan / L1 Distance
def L1(v1,v2):
    if( len(v1) != len(v2) ):
        print "error"
        return -1
    return sum([abs(v1[i]-v2[i]) for i in range(len(v1))])

#Load data
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

def zscoredata(dataarray):
    data = numpy.array(dataarray)
    N = len(data)
    #print data
    means = numpy.mean(data,axis=0)
    stds = numpy.std(data,axis=0)

    means = numpy.matlib.repmat(means, N, 1)
    stds = numpy.matlib.repmat(stds, N, 1)

    data = (data-means)/stds
    #print data
    return data

# data - data matrix, N x D where N = number of data points & D = dim of each data point
# K - number of clusters
def runkmeans(data,K):
    N = len(data) # number of data points

    # Grab 5 random points in the Dataset for random cluster means
    # rand = []
    Kmus = randsample(data, K)

    oldLabelVec = numpy.zeros(N)
    labelVec = numpy.zeros(N)
    distVector = numpy.zeros((N, K))  # Initialize matrix for distance

    count = 0
    while True:
        # calc distances from each data point to each mean
        for dataPt in range(0, N):
            for k in range(0, K):
                distVector[dataPt][k] = L1(data[dataPt], Kmus[k])
                # print(distVector[dataPt][mean])
        count += 1
        # print(count)

        # Find minimum distance, assign labels
        for d in range(0, N):
            labelVec[d] = 0
            labelVal = distVector[d][0]
            for l in range(1, K):
                if distVector[d][l] < labelVal:
                    labelVec[d] = l
                    labelVal = distVector[d][l]

                    # Check if means change/labels change.
                    # - If labels do not change, then exit loop.
        # print(oldLabelVec)
        # print 'labelVec:'
        # print (labelVec)
        if (oldLabelVec == labelVec).all():
            print("Break")
            print ('number of iterations')
            print(count)
            break
        oldLabelVec = labelVec

        # Calculate new mean based on labels
        labelCount = numpy.zeros(K)  # Count Array for values associated with a certain label
        sumVec = numpy.zeros((K, 13))  # Sum of all pts for a label.


        # print "calculating means"
        # Renu - simpler way to calc labels count and sum vector
        for x in range(0, N):
            label = int(labelVec[x])
            labelCount[label] += 1
            sumVec[label] = sumVec[label] + data[x]

        # print(sumVec)
        labelCount = labelCount[numpy.newaxis]
        labelCount = numpy.matlib.repmat(labelCount.T, 1, 13)
        # print(labelCount)
        Kmus = sumVec / labelCount
        # print(Kmus)

    return labelVec

# clusterResps - N dim array holding label of assigned cluster to nth datapoint
# externalLabels - N dim array holding external label already assigned to nth datapoint (goal characteristic from 0-4)
# K - number of clusters
def validate(clusterResps,externalLabels,K):
    results = numpy.zeros((K + 1, K + 4))
    labelVec = clusterResps
    dataLabel = externalLabels

    # format for results matrix (Confusion Matrix)

    # #######  Goal0   Goal1   Goal2   Goal3   Goal4    TOTAL     Entropy     Purity      Goal w/ max relative freq
    # Cluster1
    # Cluster2
    # Cluster3
    # Cluster4
    # Cluster5
    # TOTAL

    # count frequency in each cluster per goal
    for d in range(0, N):
        cluster = labelVec[d]
        goal = dataLabel[d]
        results[cluster][goal] += 1
    print ('calculated frequencies in results array')

    # add in totals in margins
    results[K][0:K] = numpy.sum(results[:, 0:K], axis=0)   # goal totals
    results[0:K+1, K] = numpy.sum(results[:, 0:K], axis=1)  # cluster totals

    # calc entropy


    # calc purity


    # determine goal with max relative freq per cluster
    for cluster in range(0, K):
        print results
        mode = 0  # to determine goal with max freq per cluster
        for goal in range(1, K):
            if results[cluster][goal]/results[K][goal] > results[cluster][mode]/results[K][mode]:
                mode = goal
        #results[cluster][K + 1] = numpy.sum(results[cluster])
        results[cluster][K + 3] = mode  # calc goal with max freq
        results[cluster][K + 2] = results[cluster][mode] / results[cluster][K]  # calc cluster purity
        # calc cluster entropy

    print results
    return results

def processToBinary(externalLabels):
    binaryLabels = [0 if label == 0 else 1 for label in externalLabels]
    # print binaryLabels
    return binaryLabels

def countGoals(goalLabels):
    counts = {}
    for goal in goalLabels:
        g = str(int(goal))
        if g not in counts:
            counts[g] = 1
        else:
            counts[g] += 1
    #print counts
    return counts

#print('run Kmeans with 5 clusters')
#data = zscoredata(data)
## run functions on data
#K = 5  # number of clusters
#clusterResps = runkmeans(data,K)
#validate(clusterResps,dataLabels,K)
##print dataLabels
#counts = countGoals(dataLabels)
#print counts

#print dataLabels
#print('run Kmeans with 2 clusters')
# binaryLabels = processToBinary(dataLabels)
# countGoals(binaryLabels)
# K = 2
# clusterResps = runkmeans(data,K)
# print 'assigned clusters'
# print clusterResps
# validate(clusterResps,binaryLabels,K)

print data

pca.runPCA(data)