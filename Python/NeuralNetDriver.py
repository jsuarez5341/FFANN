import numpy as np
import time
import GenNeuralNetData
import TrainNeuralNet
import NeuralNetSupportFunctions as ANNSupport

#Calls the training routine and checks the learned network parameters on the test set
def RunNeuralNetDriver(nil, maxNetworkIterations, theta, numTrain, numValidation, numTest, improveTheta):


    #See if a pretrained weight set is given
    if improveTheta:
        if theta==0:
            print("!!!ERROR: Theta is set to zero, but improveTheta is set to one. Ser improveTheta to 0 or load a valid theta weight matrix")

    #Train a network, if applicable
    if theta==0 or improveTheta:
        #Load data from MNIST
        print(">>>Loading Data From MNIST")
        trainData, validationData, testData, trainLabels, validationLabels, testLabels = GenNeuralNetData.loadMNIST(numTrain, numValidation, numTest)
        print(">>>Data Loaded")
        print(">>>Training Neural Network")
        timer=time.time()
        theta, validationErrors = TrainNeuralNet.RunTrainNeuralNet(trainData, trainLabels, validationData, validationLabels, nil, maxNetworkIterations, theta, improveTheta);
        timer=time.time()-timer
        print(">>>Training Complete. Train Time: "+str(timer) + " Seconds")
    print(">>>Testing Neural Network Accuracy")
    timer=time.time()
    #Calculate accuracy on test set
    testShape=testData.shape
    testSamples=testShape[0]
    testPixels=testShape[1]
    L=nil.size #Number of layers in FFANN

    #Initialize misclassifications
    misclassifications=0

    #Calculate predictions
    testErr=0
    for m in np.arange(testSamples):
        #Forward propagation
            activations = ANNSupport.forwardProp(testData[m, 0:testPixels], testPixels, nil, L, theta)
            #Backpropagation for final layer to calculate error
            isIncorrect=ANNSupport.checkPrediction(activations[-1, 0:nil[-1]], testLabels[m], nil, 1)
            misclassifications+=isIncorrect

    percentAccuracy = (testSamples-misclassifications)/testSamples

    timer=time.time()-timer
    print(">>>Testing Complete. Test Time: "+ str(timer) + " Seconds. Misclassified "+str(misclassifications) + " Samples. Dataset Size: " + str(testSamples) + " Samples. Test Accuracy: " + str(100*percentAccuracy) + " Percent.")
    return theta, validationErrors, percentAccuracy

