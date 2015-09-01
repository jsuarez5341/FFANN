import numpy as np

'''
This is just a collection of functions that I had to use a few times throughout my code.
Rather than copy and paste all of the code in a few places, I just threw them in another file.
The functions are: the sigmoid activation, forward propagation sequence, and evaluation of FFANN output.
'''

def hUnvec(x):
    if x>100:
        x=100
    elif x<-100:
        x=-100
    return 1/(1+np.math.exp(-x))
h=np.vectorize(hUnvec)

def forwardProp(data, dataElements, nil, L, theta):
    activations=np.zeros((L, np.max(nil)))
    activations[0, 0:nil[0]]=data[0:dataElements]
    for layer in np.arange(1, L):
        activations[layer, 0:nil[layer]] = h( np.dot(activations[layer-1, 0:nil[layer-1]],theta[0:nil[layer-1], 0:nil[layer], layer-1]) )
    return activations

def checkPrediction(predictionVec, labelVal, nil, validityOnly):
    labelVec=np.zeros(nil[-1])
    labelVec[labelVal]=1
    if validityOnly:  #Return 1 if misclassified and validityOnly (boolean requested)
        return 1 - (np.sum(labelVec==(predictionVec>0.5))==nil[-1])
    else: #Return an error vector
        errorVec = predictionVec - labelVec
        squaredError = np.dot(errorVec, errorVec.T)
        return errorVec, squaredError
