function [ trainData, trainLabels, validationData, validationLabels, testData, testLabels ] = GenNeuralNetData( setName, numTrainSamples, numValidationSamples, numTestSamples )
%GENNEURALNETDATA is mainly used to generate segments of the MNIST dataset.
%This function makes use of two external helper functions (loadMNISTImages
%and loadMNISTLabels) distributed by Stanford. The tinyMajority dataset is
%meant only for small-scale testing
%INPUT
%setName: choose 'tinyMajority' or 'MNIST'
%numTrainSamples, numValidationSamples, numTestSamples: max of 50,000 train
%samples and 5,000 of each validation and testing samples.
%OUTPUT:
%[trainData, trainLabels, validationData, validationLabels, testData, testLabels]

if strcmp('tinymajority',setName)
   data(1:2, 1:2, 1)=[1 1; 0 0];
   data(1:2, 1:2, 2)=[0 0; 1 1];
   data(1:2, 1:2, 3)=[1 0; 1 0];
   data(1:2, 1:2, 4)=[0 1; 0 1];
   data(1:2, 1:2, 5)=[1 0; 0 1];
   data(1:2, 1:2, 6)=[0 1; 1 0];
   data(1:2, 1:2, 7)=[1 1; 0 1];
   data(1:2, 1:2, 8)=[1 1; 1 0];
   data(1:2, 1:2, 9)=[1 0; 1 1];
   data(1:2, 1:2, 10)=[0 1; 1 1];
   
   data(1:2, 1:2, 11)=[1 0; 0 0];
   data(1:2, 1:2, 12)=[0 1; 0 0];
   data(1:2, 1:2, 13)=[0 0; 1 0];
   data(1:2, 1:2, 14)=[0 0; 0 1];
   data(1:2, 1:2, 15)=[0 0; 0 0];
   
   trainData=data;
   trainLabels=[ones(10,1);zeros(5,1)];
elseif strcmp('MNIST',setName)
    if numTrainSamples>50000
        numTrainSamples=50000;
    end
   
    if numValidationSamples>5000
        numValidationSamples=5000;
    end
    
    if numTestSamples>5000
        numTestSamples=5000;
    end
    
    data=   loadMNISTImages('train-images-idx3-ubyte');
    labels= loadMNISTLabels('train-labels-idx1-ubyte');

    %Data is stored with one image per column. Must reformat into 28*28 images.
    trainData=data(:, 1:numTrainSamples);
    trainData=reshape(trainData, 28, 28, numTrainSamples);
    
    validationData=data(:, 50001:50000+numValidationSamples);
    validationData=reshape(validationData, 28, 28, numValidationSamples);
    
    testData=data(:, 55001:55000+numTestSamples);
    testData=reshape(testData, 28, 28, numTestSamples);

    %Store in binary format for multiclass classification
    trainLabels=labels(1:numTrainSamples, 1);
    trainLabels = dec2bin(trainLabels)-'0';
    
    validationLabels=labels(50001:50000+numValidationSamples, 1);
    validationLabels = dec2bin(validationLabels)-'0';
    
    testLabels=labels(55001:55000+numTestSamples, 1);
    testLabels = dec2bin(testLabels)-'0';
   
end

