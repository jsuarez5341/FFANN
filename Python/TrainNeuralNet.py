import numpy as np
import NeuralNetSupportFunctions as ANNSupport

#Handles the low-level FFANN training process
def RunTrainNeuralNet(trainData, trainLabels, validationData, validationLabels, nil, maxNetworkIterations, theta, improveTheta):
    trainShape=trainData.shape
    trainSamples=trainShape[0]
    trainPixels=trainShape[1]

    validationShape=validationData.shape
    validationSamples=validationShape[0]
    validationPixels=validationShape[1]
    validationErrors=[]
    validationWeights=0
    validationLookback=2
    validationSampleRate=20

    L=nil.size #Number of layers. Note that there are L-1 layers in theta
    alpha=1 #Training constant information
    alphaTol=0.00000001

    if theta==0 and (not improveTheta):
        theta=np.zeros((np.max(nil[0:-1]), np.max(nil[1:]), L-1))
        for n in np.arange(L-1):
            #Initialize weight layers between -0.5 and 0.5
            theta[0:nil[n], 0:nil[n+1], n]=np.random.rand(nil[n], nil[n+1])-0.5

    #Initialize activation error matrix (2D).
    #Note that entries are stored row-wise in activations
    #and column-wise in deltaLC
    deltaLC=np.zeros((np.max(nil),L))

    #Begin main loop
    error=999999999
    iters=-1
    while iters<maxNetworkIterations and error>0.00001 and alpha>alphaTol:
        iters=iters+1;
        #Initialize 3D weight update matrix. Must be done each iteration
        deltaUC=np.zeros(theta.shape)

        #Reset additional iteration specific variables
        error=0
        print("Iterations: "+str(iters)) #Useful feedback for long training sessions

        for m in np.arange(trainSamples):
            #Forward propagation
            activations = ANNSupport.forwardProp(trainData[m, 0:trainPixels], trainPixels, nil, L, theta)

            #Backpropagation
            #Output layer is easy but must be done separately
            errorVec, squaredError=ANNSupport.checkPrediction(activations[-1, 0:nil[-1]], trainLabels[m], nil, 0)
            deltaLC[0:nil[L-1],L-1]=errorVec.T
            deltaUC[0:nil[-2],0:nil[-1],-1] += np.outer(activations[-2, 0:nil[-2]], deltaLC[0:nil[-1],-1])
            #Harvest squared error in final layer
            error += squaredError
            #Hidden layers. No update to input layer (no error)
            for layer in np.arange(L-2, 0, -1):
                activationsInPrevious = activations[layer, 0:nil[layer]]
                deltaLC[0:nil[layer], layer] = np.dot(theta[0:nil[layer],0:nil[layer+1],layer],deltaLC[0:nil[layer+1],layer+1]).T * activationsInPrevious * (1-activationsInPrevious)
                deltaUC[0:nil[layer-1],0:nil[layer],layer-1] += np.outer(activations[layer-1, 0:nil[layer-1]], deltaLC[0:nil[layer],layer])

        #Scale weights by layer
        for layer in np.arange(L-1):
            tempDeltaUCLayer = deltaUC[0:nil[layer],0:nil[layer+1], layer]
            deltaUC[0:nil[layer],0:nil[layer+1],layer] = tempDeltaUCLayer/np.max(np.abs(tempDeltaUCLayer))

        #Update theta. Error always decreases in batch mode
        done=0
        while not done:
            newTheta = theta - alpha*deltaUC
            newError=0
            for m in np.arange(trainSamples):
                #Forward propagation
                activations = ANNSupport.forwardProp(trainData[m, 0:trainPixels], trainPixels, nil, L, newTheta)
                #Backpropagation for final layer to calculate error
                errorVec, squaredError=ANNSupport.checkPrediction(activations[-1, 0:nil[-1]], trainLabels[m], nil, 0)
                newError += squaredError

            #Implementation of momentum
            if newError<error:
                theta=newTheta;
                alpha*=2;
                error=newError;
                done=1;
                print('new error = '+str(error))
                print('alpha = '+str(alpha));
            else: #Error has increased
                alpha/=2

        #Evaluation of error on validation set
        if validationSamples>0 and iters%validationSampleRate==0:
            validationErr=0
            for m in np.arange(validationSamples):
                #Forward propagation
                activations = ANNSupport.forwardProp(validationData[m, 0:validationPixels], validationPixels, nil, L, theta)
                #Backpropagation for final layer to calculate error
                squaredError=ANNSupport.checkPrediction(activations[-1, 0:nil[-1]], validationLabels[m], nil, 1)
                validationErr+=squaredError

            #Validation errors should be saved for early stopping and plotting
            validationErrors.append(validationErr)

            #Save the set of weights that performs better on the validations set
            if len(validationErrors)>1 and validationErrors[-1]<validationErrors[-2]:
                    validationWeights=theta

            #Check for overfitting
            if len(validationErrors)>validationLookback and np.max(validationErrors[-validationLookback:])==validationErrors[-1]:
                #Stop execution
                print("Overfitting Detected. Returning Weights")
                break

    #End of outer iteration loop
    if iters==maxNetworkIterations:
        print('Max Iterations Reached. Returning Weights')
    elif error<=0.00001:
        print('Error Tolerance Reached')
    elif alpha<=alphaTol:
        print('Learning Rate Tolerance Reached')

    if np.sum(validationWeights)==0:
        print('Training Weights Returned')
        return theta, validationErrors
    else:
        print('Validation Weights Returned')
        return validationWeights, validationErrors



