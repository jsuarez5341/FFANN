#MAIN PROGRAM FILE

'''
Easy FFANN creation: specify config-style parameters
in this file. No other modifications are required.
TrainNeuralNet.py contains additional parameters
that only advanced users should modify, as they are
more technical and pertain only to the validation
set. This implementation trains via backpropagation,
uses momentum based error decent, and also contains
validation set checking for early stopping to prevent
overfitting (a plot will be displayed after code execution.
This will show the validation error and allow the user
to ensure that no overfitting is occurring)
This code is also written such that the user can load a
pre-trained 3D array for fine tuning, but will have to
write their own (very simple) loading scheme.
'''

import numpy as np
import matplotlib.pylab as mpl
import time

import NeuralNetDriver

if __name__ == '__main__':
    #Number of train/validation/test samples
    numTrain=59900
    numValidation=100
    numTest=2000

    #Driver parameters
    nil=np.array([784, 10])
    maxNetworkIterations=400
    theta=0
    improveTheta=0


    #Start the FFANN driver
    print('***Starting Neural Net Driver')
    timer=time.time()
    theta, validationErrors, percentAccuracy = NeuralNetDriver.RunNeuralNetDriver(nil, maxNetworkIterations, theta, numTrain, numValidation, numTest, improveTheta)
    timer=time.time()-timer
    print('***Driver Execution Complete. Total Runtime: '+str(timer)+" Seconds.")
    #Note: the x-axis is only a general indication of amount of training. This is because of the manner in which validation error is sampled.
    print('***Plotting Validation Data. Y-Axis = Number Misclassifications')
    mpl.plot(validationErrors)
    mpl.show()