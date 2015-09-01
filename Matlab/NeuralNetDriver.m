function [ theta, missclassifications, validationTheta, validationErrors] = NeuralNetDriver(nil, maxNetworkIterations, theta, numTrainingSamples, numValidationSamples, numTestingSamples, improveTheta)
%NEURALNETDRIVER calls TrainNeuralNetwork and evaluates the resulting
%parameters on a testing set. The MNIST handwritten digit dataset will be
%used for training, validation, and testing.
%INPUT: See TRAINNEURALNet
%nil, maxNetworkIterations, theta, numTrainingSamples, numValidationSamples, numTestingSamples, improveTheta
%OUTPUT:
%[theta, missclassifications, validationTheta, validationErrors]

%Generate data and labels from MNIST
[trainData, trainLabels, validationData, validationLabels, testData, testLabels] = GenNeuralNetData('MNIST', numTrainingSamples, numValidationSamples, numTestingSamples);

if isnan(theta) || improveTheta
    %Train a neural network and return theta (the 3D weight matrix)
    [validationTheta, theta, validationErrors] = TrainNeuralNet(trainData, trainLabels, validationData, validationLabels, nil, maxNetworkIterations, theta, improveTheta);
end

%Calculate accuracy on the testing set
[dataWidth, dataHeight, numSamples]= size(testData);
L=length(nil);

%Sigmoid
h=@(x) 1./(1+exp(-x));

%Number of sampled incorrectly classified
missclassifications=0;

for m=1:numSamples
    %Forward Propagation. No need to store activation matrix.
    tempActivationVec = reshape(testData(:, :, m), 1, dataHeight.*dataWidth); %Unroll data;
    for layer = 2:L
        tempActivationVec =  h(tempActivationVec*validationTheta(1:nil(layer-1), 1:nil(layer), layer-1));
    end
    
    %Backpropagation for first layer only (find errors)
    missclassifications = missclassifications + sum(unique(abs(tempActivationVec - testLabels(m, 1:nil(L))) >= .5));
end

end

