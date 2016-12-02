################################# IMPORT #######################################
from pprint import pprint
from random import randint
from random import sample as randsample
from decimal import Decimal
import numpy
import numpy.matlib
import pickle
#################################### FUNCTIONS ################################
def getoccurence(s,h):
    return [i for i, letter in enumerate(s) if letter == h]
#Manhattan / L1 Distance
def L1(v1,v2):
    if( len(v1) != len(v2) ):
        print "error"
        return -1
    return sum([abs(v1[i]-v2[i]) for i in range(len(v1))])

#MAIN#
########################## Processing Data for our use #########################
# f = open("processed.cleveland.data","r")
# #Save Data into usable file type, first to array, then pickle array.
# #Note: We changed ? to 0.0 values
# data = []
# c = 0
# for lines in f:
#     commInd = getoccurence(lines,',')
#     data.append([])
#
#     for i in range(0,len(commInd)):
#         if i == 0:
#             data[c].append(Decimal(lines[:commInd[i]]))
#         elif i == len(commInd)-1:
#             data[c].append(Decimal(lines[commInd[i-1]+1:commInd[i]]))
#             data[c].append(Decimal(lines[commInd[i]+1:].strip('\n')))
#         else:
#             data[c].append(Decimal(lines[commInd[i-1]+1:commInd[i]]))
#     c = c + 1
# # pprint(data)
# pickle.dump(data, open("ClevelandProcData.p", "wb"))
################################ K MEANS CODE #################################

#Load data
data = pickle.load(open("ClevelandProcData.p","rb"))
N = len(data)  # number of data points
K = 5          # number of clusters
print N

#Turn all decimal data into float data
for a in range(0, N):
    for b in range(0,len(data[a])):
        data[a][b] = float(data[a][b])
#Seperate data from label for crossvalidation
dataLabel = [row[13] for row in data]
for dat in range(0, N):
    del data[dat][13]
#print (dataLabel)
#pprint(data)

#Grab 5 random points in the Dataset for random cluster means
# rand = []
rand = randsample(data, K)

# for i in range(0,5):
#     random = randint(0,len(data))
#     rand.append(data[random])

#print(rand)
#print(len(data[random]))
# - Create a label array as a result

oldLabelVec = numpy.zeros(N)
labelVec = numpy.zeros(N)
distVector = numpy.zeros((N, K)) #Initialize matrix for distance
#Loop This:
    #Find Closest datapoints to those randomized Points
        # - Distance metric or PCA
        # - Determine label of points via lowest Distance
count = 0
while count <1:
    # calc distances from each data point to each mean
    for dataPt in range(0, N):
        for k in range(0,K):
            distVector[dataPt][k] = L1(data[dataPt], rand[k])
            #print(distVector[dataPt][mean])
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
    #print(oldLabelVec)
    print 'labelVec:'
    print (labelVec)
    if (oldLabelVec==labelVec).all():
        print("Break")
        break
    oldLabelVec = labelVec

    # Calculate new mean based on labels
    labelCount = numpy.zeros(K) # Count Array for values associated with a certain label
    meanVec = numpy.zeros((5,13)) # Sum of all pts for a label.
    # for z in range(0,5):
    #     for y in range(0,len(labelVec)):
    #         if labelVec[y] == z:
    #             labelCount[z] += 1
    #             #print("MeanVec[z]: %s" %meanVec[z])
    #             #print("data[y]: %s" %data[y])
    #         for x in range(0,13): # Iterate through each dimension of vector
    #             meanVec[z][x] += data[y][x]

    print "calculating means"
    # Renu - simpler way to calc labels count and sum vector
    for x in range(0, N):
        label = int(labelVec[x])
        labelCount[label] += 1
        meanVec[label] = meanVec[label] + data[x]

    #print(meanVec)
    labelCount = labelCount[numpy.newaxis]
    labelCount = numpy.matlib.repmat(labelCount.T,1,13)
    #print(labelCount)
    rand = meanVec/labelCount
    #print(rand)

################################# CROSS VALIDATION ################################
#Cross validate with data that we have.
loss = 0
for q in range(0, N):
    if labelVec[q] != dataLabel[q]:
        loss = loss + 1

print(loss)


################################# EXTERNAL VALIDATION ################################
# count for each cluster, how many data points fall into each goal (OG labels = dataLabel)
results = numpy.zeros((K+1, K+2)) # rows are determined clusters, columns are OG goals from data Label & entropy & purity & max

# #######  Goal0   Goal1   Goal2   Goal3   Goal4    Entropy     Purity      Goal w/ max freq
# Cluster1
# Cluster2
# Cluster3
# Cluster4
# Cluster5
# TOTAL

# count frequency in each cluster per goal
for d in range(0,N):
    cluster = labelVec[d]
    goal = dataLabel[d]
    results[cluster][goal] += 1
# add in totals
for k in range(0,K):
    results[K][k] = numpy.sum(results[0:K][k])
#print results
#print labelCount

# Graph it to visualize data:
