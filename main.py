# This is a sample Python script.
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#The general idea of this paper is that we have a known error rate within our sample. Therefore,
#this will be provided by the user when starting the program.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import csv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def chooseRunMode():
    # Use a breakpoint in the code line below to debug your script.
    run_type = int(input("Linear (1) or Random Data? (2) "))
    noise1 = float(input("Enter noise level for Class 0 (0->1) "))
    noise2 = float(input("Enter noise level for Class 1 (0->1) "))
    return run_type, noise1, noise2

def generateLinearData():
    print("Generating Linear Data")
    X = np.random.rand(2000,2)*100
    y_size = 1400
    Y = -1*np.ones((y_size,2))
    index = 0
    for i in range(len(X)):
        #I am making a new array, Y, that does not have any values near the middle,
        #So there is clearly 2D separated data
        if not((X[i,1] > (X[i,0] - 8)) and (X[i,1] < (X[i,0] + 8))):
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
    Class0 = np.zeros((1,2))
    Class1 = np.zeros((1,2))

    with open('banana.csv', 'r') as file:
        reader = csv.reader(file,delimiter=',', quotechar='"')
        next(file)
        for row in reader:
            x1 = float(row[0])
            x2 = float(row[1])
            if(row[2] == '1.0'):
                Class1 = np.vstack([Class1,[x1,x2]])
            else:
                Class0 = np.vstack([Class0,[x1,x2]])
    print(Class1)
    return Class0, Class1
def generateNoisyData(noise0, noise1, Class0, Class1):
    print("Generating Noisy Data -- flipping data")
    #First thing to do, is create a new matrix, of 3 dimension now, with the third dimension
    #being 1 or -1 based on the class.
    Class0_y = -1*np.ones((len(Class0),1))
    Class0 = np.hstack([Class0, Class0_y])
    Class1_y = 1*np.ones((len(Class1),1))
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
                combinedData[i,2] = 1
        if(flip1counter < len(flip1index) ):
                if i == flip1index[flip1counter]:
                    combinedData[i,2] = -1
                    flip1counter = flip1counter + 1
        if combinedData[i,2] == 1:
            plt.scatter(combinedData[i,0], combinedData[i,1], c='r', label='Class1' )
        else:
            plt.scatter(combinedData[i,0], combinedData[i,1], c='b', label='Class0')
    plt.title("Noise Added to Linearly Separable Data")
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

    print("Checkpoint 3: Split Data into Train & Test")
    return trainData, testData

def trainAlgorithm(trainData, noise0, noise1, Class0, Class1):
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
    pi_0 = len(class0_train)/(len(class1_train)+len(class0_train))
    pi_1 = len(class1_train)/(len(class1_train)+len(class0_train))

    x0 = trainData[:,0]
    x1 = trainData[:,1]
    naive_prediction = np.zeros((len(trainData[:,0]), 3))
    good_prediction = np.zeros((len(trainData[:,0]), 3))

    for i in range(0, np.size(x0)):
            block = np.zeros((2, 1))
            block[0] = x0[i]
            block[1] = x1[i]

            #unlike in the example from Homework 3 which was an 8x8 block, we can just do a 2x1 block.
            guess1 = -1 / 2 * np.matmul(np.matmul(np.transpose(block - mu1), inv_sigma1), (block - mu1)) + np.log(
                pi_1) - 1 / 2 * np.log(det_sigma1)
            guess0 = -1 / 2 * np.matmul(np.matmul(np.transpose(block - mu0), inv_sigma0), (block - mu0)) + np.log(
                pi_0) - 1 / 2 * np.log(det_sigma0)
            # print(str(guess1 > guess0) + " " + str(guess0) + " " + str(guess1) + " " + str(x0[i]) + " " + str(x0[j]))

            naive_prediction[i,0] = x0[i]
            naive_prediction[i,1] = x1[i]
            if guess1 > guess0:
                naive_prediction[i,2] = 1
            else:
                naive_prediction[i,2] = -1

            guess1_good =  -1 / 2 * np.matmul(np.matmul(np.transpose(block - mu1), inv_sigma1), (block - mu1)) + np.log(
                pi_1) - 1 / 2 * np.log(det_sigma1) - 1/2*np.log(2*np.pi)
            guess0_good = -1 / 2 * np.matmul(np.matmul(np.transpose(block - mu0), inv_sigma0), (block - mu0)) + np.log(
                pi_0) - 1 / 2 * np.log(det_sigma0) - 1/2*np.log(2*np.pi)
            #This is my attempt at incorporating the logic from the paper.
            good_prediction[i, 0] = x0[i]
            good_prediction[i, 1] = x1[i]
            good_guess = np.exp(guess1_good)/(np.exp(guess0_good)+np.exp(guess1_good))
            #This if statement is saying P(x given y = 1) * P(Y = 1) / ( P( x given y = 1) * P(y=1) + P(x given y = -1)*P(y=-1)
            if np.sign(good_guess - (0.5 - noise0)/(1-noise1-noise0)) == 1:
                good_prediction[i,2] = 1
            else:
                good_prediction[i,2] = -1

            if naive_prediction[i,2] != good_prediction[i,2]:
                print(naive_prediction[i,:])
                print(str(guess1) + " " + str(guess0))
                print(str(guess1_good) + " " + str(guess0_good) + str(good_guess))


    plt.figure(4)
    for i in range(0, (len(trainData[:,0]))):
        if naive_prediction[i,2] == 1:
            plt.scatter(naive_prediction[i, 0], naive_prediction[i, 1], c='r')
        else:
            plt.scatter(naive_prediction[i, 0], naive_prediction[i, 1], c='b')

    plt.title('Data with Naive Boundary Drawn')
    plt.show()
    #plt.savefig('naive.png')

    plt.figure(5)
    for i in range(0, (len(trainData[:,0]))):
        if good_prediction[i,2] == 1:
            plt.scatter(good_prediction[i, 0], good_prediction[i, 1], c='r')
        else:
            plt.scatter(good_prediction[i, 0], good_prediction[i, 1], c='b')
    plt.title('Data with Good Boundary Drawn')
    plt.show()
    #plt.savefig('good.png')

    return naive_prediction, good_prediction

def evaluateLinear(naive, good):
    #I am evaluating based on knowing how I split the data
    incorrect_naive = 0
    incorrect_good = 0

    for i in range(0, len(naive)):
        if ((naive[i, 0] - naive[i, 1]) < 0 and (naive[i,2] == 1)) or ((naive[i,0] - naive[i,1] > 0) and naive[i,2] == -1):
            incorrect_naive = incorrect_naive + 1
        if (good[i, 0] - good[i, 1]) < 0 and (good[i, 2] == 1) or ((good[i,0] - good[i,1] > 0) and good[i,2] == -1):
            incorrect_good = incorrect_good + 1

    print("Total Number of Data Points tested:" + str(len(good)))
    print("Incorrect naive guess: " + str(incorrect_naive) + " Percent correct: " + str((len(naive)-incorrect_naive)/len(naive)))
    print("Incorrect good guess: " + str(incorrect_good) +  " Percent correct: " + str((len(good)-incorrect_good)/len(good)))
if __name__ == '__main__':
    runMode, noise0, noise1 = chooseRunMode() #Choose synthetic data or random
    if runMode == 1:
        Class0, Class1 = generateLinearData() #generate synthetic linearly separable 2-D data.
    if runMode == 2:
        Class0, Class1 = generateRandomData() #using "banana" dataset like in the actual paper. 

    noisyData = generateNoisyData(noise0,noise1, Class0,Class1) #modify for noise rates given
    trainData, testData = splitData(noisyData)

    naive, good = trainAlgorithm(trainData, noise0, noise1, Class0, Class1)
    if runMode == 1:
        evaluateLinear(naive, good)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
