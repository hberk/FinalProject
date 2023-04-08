# This is a sample Python script.
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#The general idea of this paper is that we have a known error rate within our sample. Therefore,
#this will be provided by the user when starting the program.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
def chooseRunMode():
    # Use a breakpoint in the code line below to debug your script.
    run_type = int(input("Linear (1) or Random Data? (2) "))
    noise1 = float(input("Enter noise level for Class 0 (0->1) "))
    noise2 = float(input("Enter noise level for Class 1 (0->1) "))
    return run_type, noise1, noise2

def generateLinearData():
    print("Generating Linear Data")
    X = np.random.rand(1000,2)*100
    y_size = 650
    Y = -1*np.ones((y_size,2))
    index = 0
    for i in range(len(X)):
        #I am making a new array, Y, that does not have any values near the middle,
        #So there is clearly 2D separated data
        if not((X[i,1] > (X[i,0] - 10)) and (X[i,1] < (X[i,0] + 10))):
            if(index < y_size):
                Y[index,0] = X[i,0]
                Y[index,1] = X[i,1]
                index = index + 1

    #Now that I have y, I can split it into its Class 0 and Class 1 properly.
    Class0 = []
    Class1 = []
    for i in range(len(Y)):
        if(Y[i,0] - Y[i,1] ) < 0:
            #plt.scatter(Y[i, 0], Y[i, 1], c='b')
            Class0.append([Y[i,0], Y[i,1]])
        else:
            #plt.scatter(Y[i, 0], Y[i, 1], c='r')
            Class1.append([Y[i, 0], Y[i, 1]])

    print("Checkpoint 1: Synthetic 2-D Linearly Separable Data Created")
    #plt.show()
    return Class0, Class1

def generateRandomData():
    print("Generating Random Data")

def generateNoisyData(noise0, noise1, Class0, Class1):
    print("Generating Noisy Data -- flipping data")
    #First thing to do, is create a new matrix, of 3 dimension now, with the third dimension
    #being 1 or -1 based on the class.
    Class0_y = np.ones((len(Class0),1))
    Class0 = np.hstack([Class0, Class0_y])
    Class1_y = np.zeros((len(Class1),1))
    Class1 = np.hstack([Class1, Class1_y])

    combinedData = np.vstack([Class0, Class1])
    #plt.scatter(combinedData[:,0], combinedData[:,1])
    #plt.show()

    #Given noise1 rate, flip necessary amount of data in Class0.
    flip0amount = int(len(Class0)*noise0)
    flip0index = np.random.rand(flip0amount,1)*len(Class0)
    flip0index = np.round(flip0index).astype(int)
    flip0index = np.sort(flip0index, axis = None)
    flip0index = np.unique(flip0index)

    flip1amount = int(len(Class1)*noise1)
    flip1index = np.random.rand(flip1amount,1)*len(Class1)
    flip1index = np.around(flip1index).astype(int) + len(Class0)
    flip1index = np.sort(flip1index, axis = None)
    flip1index = np.unique(flip1index)

    flip0counter = 0
    flip1counter = 0
    for i in range(len(combinedData)):
        if(flip0counter < len(flip0index)):
            if i == flip0index[flip0counter]:
                flip0counter = flip0counter + 1
                combinedData[i,2] = -1
        if(flip1counter < len(flip1index) ):
                if i == flip1index[flip1counter]:
                    combinedData[i,2] = 1
                    flip1counter = flip1counter + 1
        if combinedData[i,2] == 1:
            plt.scatter(combinedData[i,0], combinedData[i,1], c='r', label='Class1' )
        else:
            plt.scatter(combinedData[i,0], combinedData[i,1], c='b', label='Class0')
    plt.show()
    print("Checkpoint 2: Random Noise Inserted into Linearly Separable Data")
    return combinedData
# Press the green button in the gutter to run the script.

def splitData(noisyData):
    #The purpose of this function is to split the data into train data and test data.
    #The ratio will be about 80% train, 20% test.

    trainData = [[0,0,0]]
    testData = [[0,0,0]]
    for i in range(len(noisyData)):
        k = np.random.rand()
        if k < 0.8:
            trainData = np.vstack([trainData,noisyData[i,:]])
        else:
            testData = np.vstack([testData,noisyData[i,:]])

    # plt.figure(2)
    # plt.scatter(trainData[:,0], trainData[:,1])
    # plt.show()
    # plt.figure(3)
    # plt.scatter(testData[:,0], testData[:,1])
    # plt.show()
    print("Checkpoint 3: Split Data into Train & Test")
    return trainData, testData

def trainAlgorithm(trainData, noise0, noise1):
    print("Starting to train the model....")
    #First need to find the mean & StdDev of each DataSet for Gaussian
    class1_count = np.count_nonzero(trainData[:,2], axis=0)
    class0_count = len(trainData) - class1_count
    class0_train = np.zeros((1,3))

    class1_train = np.zeros((1,3))

    for i in range(0,len(trainData)):
        if(trainData[i,2] == 1):
            class1_train = np.vstack([class1_train, trainData[i,:]])
        else:
            class0_train = np.vstack([class0_train, trainData[i,:]])

    class0_train = class0_train[2:len(class0_train),:]
    class1_train = class1_train[2:len(class1_train), :]
    mu0 = np.zeros((2,1))
    mu1 = np.zeros((2, 1))
    mu0[0] = np.average(class0_train[:,0])
    mu0[1] = np.average(class0_train[:,1])
    mu1[0] = np.average(class1_train[:,0])
    mu1[1] = np.average(class1_train[:,1])
    print("Average for class0: " + str(mu0))
    print("Average for class1: " + str(mu1))
    class0_sigma = [[0, 0], [0, 0]]
    class1_sigma = [[0, 0], [0, 0]]

    for i in range(0,2):
        for j in range(0,2):
            col0a = class0_train[:, i]
            col0b = class0_train[:, j ]
            class0_sigma[i][j] = np.cov(col0a, col0b)[i][j]
            col1a = class1_train[:, i ]
            col1b = class1_train[:, j]
            class1_sigma[i][j] = np.cov(col1a, col1b)[i][j]

    det_sigma0 = np.linalg.det(class0_sigma)
    det_sigma1 = np.linalg.det(class1_sigma)
    inv_sigma0 = np.linalg.inv(class0_sigma)
    inv_sigma1 = np.linalg.inv(class1_sigma)
    pi_0 = len(class1_train)/(len(class1_train)+len(class0_train))
    pi_1 = len(class0_train)/(len(class1_train)+len(class0_train))

    x0 = np.linspace(0, 100, 200)
    x1 = np.linspace(0, 100, 200)
    naive_prediction = np.zeros((200, 200))
    good_prediction = np.zeros((200,200))
    for i in range(0, 200):
        for j in range(0, 200):
            block = np.zeros((2, 1))
            block[0] = x0[i]
            block[1] = x1[j]

            guess1 = -1 / 2 * np.matmul(np.matmul(np.transpose(block - mu1), inv_sigma1), (block - mu1)) + np.log(
                pi_1) - 1 / 2 * np.log(det_sigma1)
            guess0 = -1 / 2 * np.matmul(np.matmul(np.transpose(block - mu0), inv_sigma0), (block - mu0)) + np.log(
                pi_0) - 1 / 2 * np.log(det_sigma0)
            # print(str(guess1 > guess0) + " " + str(guess0) + " " + str(guess1) + " " + str(x0[i]) + " " + str(x0[j]))

            if guess1 > guess0:
                naive_prediction[i][j] = 1

            else:
                naive_prediction[i][j] = -1

            #print(guess1)
            #print(guess1 - (0.5 - noise0)/(1-noise1-noise0))
            print(np.exp(guess1)/(np.exp(guess0)+np.exp(guess1)))
            if np.sign(np.exp(guess1)/(np.exp(guess0)+np.exp(guess1)) - (0.5 - noise0)/(1-noise1-noise0)) == 1:
                good_prediction[i][j] = 1
            else:
                good_prediction[i][j] = -1
    col = []
    for i in range(0,200):
        for j in range(0,200):
            if(naive_prediction[i][j]==1):
                col.append('r')
            else:
                col.append('b')
    X, Y = np.mgrid[0:100:200j, 0:100:200j]
    plt.figure(3)
    plt.scatter(X,Y,c=col) #comment
    plt.show()
    col = []
    for i in range(0, 200):
        for j in range(0, 200):
            if (good_prediction[i][j] == 1):
                col.append('r')
            else:
                col.append('b')
    X, Y = np.mgrid[0:100:200j, 0:100:200j]
    plt.figure(3)
    plt.scatter(X, Y, c=col)
    plt.show()
    X, Y = np.mgrid[0:100:200j, 0:100:200j]
    prediction = naive_prediction.reshape(X.shape)
    plt.figure(4)
    plt.contour(X, Y, prediction, levels=[0])
    plt.scatter(class0_train[:, 0], class0_train[:, 1], c='b')
    plt.scatter(class1_train[:, 0], class1_train[:, 1], c='r')
    plt.legend(['Class0', 'Class1', 'Boundary'])
    plt.title('Data with Naive Boundary Drawn')
    plt.savefig('naive.png')

    prediction = good_prediction.reshape(X.shape)
    plt.figure(5)
    plt.contour(X, Y, prediction, levels=[0])
    plt.scatter(class0_train[:, 0], class0_train[:, 1], c='b')
    plt.scatter(class1_train[:, 0], class1_train[:, 1], c='r')
    plt.legend(['Class0', 'Class1', 'Boundary'])
    plt.title('Data with Good Boundary Drawn')
    plt.savefig('good.png')

def lossFunction(x,y):
    if x == y:
        return 0
    else:
        return 1

def estimatedLoss(x,y,prob0, prob1, noise0,noise1):
    loss = ((1-prob0)*lossFunction(x,y) - (prob1)*lossFunction(x,y))/(1-prob0-prob1)


if __name__ == '__main__':
    runMode, noise0, noise1 = chooseRunMode() #Choose synthetic data or random
    if runMode == 1:
        Class0, Class1 = generateLinearData() #generate synthetic linearly separable 2-D data.
    if runMode == 2:
        Class0, Class1 = generateRandomData()

    noisyData = generateNoisyData(noise0,noise1, Class0,Class1) #modify for noise rates given
    trainData, testData = splitData(noisyData)

    trainAlgorithm(trainData, noise0, noise1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
