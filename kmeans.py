################################# IMPORT #######################################
from pprint import pprint
from random import randint
from random import sample
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

#Turn all decimal data into float data
for a in range(0,len(data)):
    for b in range(0,len(data[a])):
        data[a][b] = float(data[a][b])
#Seperate data from label for crossvalidation
dataLabel = [row[13] for row in data]
for dat in range(0,len(data)):
    del data[dat][13]
#print (dataLabel)
#pprint(data)

#Grab 4 random points in the Dataset for random cluster means
# rand = []
rand = sample(data, 4)

# for i in range(0,5):
#     random = randint(0,len(data))
#     rand.append(data[random])



#print(rand)
#print(len(data[random]))
# - Create a label array as a result
oldLabelVec = numpy.zeros(len(data))
labelVec = numpy.zeros(len(data))
distVector = numpy.zeros((len(data),5)) #Initialize matrix for distance
#Loop This:
    #Find Closest datapoints to those randomized Points
        # - Distance metric or PCA
        # - Determine label of points via lowest Distance
count = 0
while True:
    for dataPt in range(0,len(data)):
        for mean in range(0,5):
            distVector[dataPt][mean] = L1(data[dataPt],rand[mean])
            #print(distVector[dataPt][mean])
    count += 1
    print(count)
    for d in range(0,len(distVector)):
        label = 0
        labelVal = 0
        for l in range(0,5):
            if l == 0:
                labelVal = distVector[d][l]
            else:
                if distVector[d][l] < labelVal:
                    label = l
                    labelVal = distVector[d][l]
        labelVec[d] = label
    #Check if means change/labels change.
        # - If labels do not change, then exit loop.
    print(oldLabelVec)
    print(labelVec)
    if (oldLabelVec==labelVec).all():
        print("Break")
        break
    else:
        oldLabelVec = labelVec
    #Calculate new mean based on label
    labelCount = numpy.zeros(5) # Count Array for values associated with a certain label
    meanVec = numpy.zeros((5,13)) # Sum of all pts for a label.
    for z in range(0,5):
        for y in range(0,len(labelVec)):
            if labelVec[y] == z:
                labelCount[z] += 1
                #print("MeanVec[z]: %s" %meanVec[z])
                #print("data[y]: %s" %data[y])
            for x in range(0,13): # Iterate through each dimension of vector
                meanVec[z][x] += data[y][x]
    #print(meanVec)
    labelCount = labelCount[numpy.newaxis]
    labelCount = numpy.matlib.repmat(labelCount.T,1,13)
    #print(labelCount)
    rand = meanVec/labelCount
    print(rand)
################################# CROSS VALIDATION ################################
#Cross validate with data that we have.
loss = 0
for q in range(0,len(data)):
    if labelVec[q] != dataLabel[q]:
        loss = loss + 1

print(loss)

#Graph it to visualize data:
