%NEURALNETSCRIPT provides an easy way to train a feed-forward artificial
%neural network via backpropagation on the popular MNIST dataset. This file
%allows for easy setting of all parameters nessesary to tweak the network.
%Note that using large numbers of samples with large neural networks will
%cause code to run slowly (days or weeks). The amount of work required to
%gain increasingly high accuracy on the dataset increases rapidly. Also
%note that the MNIST dataset has 60,000 samples. A maximum of 50,000 may be
%used for training and a maximum of 5,000 for each validation and testing.
%The neural network intentionally outputs data to the command window for
%progress monitoring on longer runs.

%Cleanup & Setup
clear;
clc;
validationTheta=nan;

%--------------------------------------------------------------------------
%Uncomment and edit the following line to use a predefined value of theta.
%Note that parameters must be set to match any theta uploaded.
%load('validation.mat');

%Parameters to edit
nil=                        [784, 4];
maxNetworkIterations=       100;
numTrainingSamples=         2000;
numValidationSamples=       2000;
numTestingSamples=          2000;
improveTheta=               0;

[theta, missclassifications, validationTheta, validationErrors] = ...
    NeuralNetDriver( nil, maxNetworkIterations, validationTheta, numTrainingSamples, ...
    numValidationSamples, numTestingSamples, improveTheta);


%Display percent accuracy
display(horzcat('Percent Accuracy: ',...
    num2str(100.*(numTestingSamples-missclassifications)./numTestingSamples)));

%SPECIFY FILE NAMES BEFORE RUNNING CODE. BE CAREFUL NOT TO OVERWRITE.
display('Type "return" to save variables');
save('FFANN_DATA.mat');
save('validationTheta.mat','validationTheta');