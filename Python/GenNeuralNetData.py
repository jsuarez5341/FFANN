'''This file is used to load the MNIST dataset.
It leverages MNISTReader (citation included in the respective
.py file) to load data, and then creates train/validation/test
sets based on user input.'''

import MNISTReader
import numpy as np

def loadMNIST(numTrain, numValidation, numTest):
    dataReader=MNISTReader.MNIST()
    trainData, trainLabels=dataReader.load_training()
    testData, testLabels=dataReader.load_testing()
    #Note: there are 60,000 samples in trainData and
    #10,000 in testData to be partitioned
    if numTrain>60000:
        print("There are only 60,000 samples in the training set. Using 60,000 samples.")
        numTrain=60000
    if numTrain+numValidation>60000:
        print("There are only enough samples for a TOTAL of 60,000 training and validation samples. Will use "+str(60000-numTrain)+" validation samples.")
        numValidation=60000-numTrain
    if numTest>10000:
        print("There are only 10,000 samples in the testing set. Using 10,000 samples")
        numTest=10000
    trainValidationData=np.asarray(trainData)/255
    trainData=trainValidationData[0:numTrain, 0:784]
    validationData=trainValidationData[numTrain:numTrain+numValidation, 0:784]
    testData=np.asarray(testData)/255
    testData=testData[0:numTest, 0:784]

    trainValidationLabels=np.asarray(trainLabels)
    trainLabels=trainValidationLabels[0:numTrain]
    validationLabels=trainValidationLabels[numTrain:numTrain+numValidation]
    testLabels=np.asarray(testLabels)
    testLabels=testLabels[0:numTest]
    return trainData, validationData, testData, trainLabels, validationLabels, testLabels