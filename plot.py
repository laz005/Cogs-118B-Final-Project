import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pickle
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


def plotGoalVsMean(dataLabels, data):
    #For each of 13 dimensions, get means for each goal.
    ATTRIBUTES = ['Age','Sex','Chest Pain Type','Resting Blood Pressure',
        'Cholesterol Level(mg/dl)','Fasting Blood Sugar(>mg/dl), 1=yes, 2 = no'
        ,'Resting ECG',"Maximum Heart Rate","Exercise induced angina (1=yes 2=no)",
        'ST depression induced by exercise relative to rest',"slope of peak exercise ST segment",
        'Number of major vessels colored', 'Thal (defect)' ]
    meanData = np.zeros((5,13))
    count = np.zeros(5)
    for j in range(0,5):
        for i in range(0,len(dataLabels)):
            if dataLabels[i] == j:
                meanData[j] = np.add(meanData[j],data[i])
                count[j] += 1
        meanData[j] = meanData[j]/np.matlib.repmat(count[j], 1, 13)
        print(meanData)
        print(count)
    f, axarr = plt.subplots(5,3)
    for k in range(0,13):
        x = np.arange(0,5,1)
        y = meanData[:,k]
        axarr[k/3,k%3].plot(x,y,'ro')
        title = ATTRIBUTES[k]
        axarr[k/3,k%3].set_title(title)
    plt.show()
plotGoalVsMean(dataLabels,data)
