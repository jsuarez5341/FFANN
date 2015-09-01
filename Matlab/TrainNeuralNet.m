function [ validationWeights, weights, validationErrors] = TrainNeuralNet( trainData, trainLabels, validationData, validationLabels, nil, maxNetworkIterations, theta, improveTheta)
%TRAINNEURALNET trains a feed-forward artificial neural network using
%backpropagation learning. The network automatically stops training when
%performance decreases on the validation set.
%INPUT:
%trainData: 3D matrix containing greyscale training images
%trainLabels: 2D matrix containing one, binary label per row
%validationData: a separate subset of data used to predict network accuracy
%validationLabels: 2D matrix containing one, binary label per row
%nil: neurons-in-layer. Vector containing the number of neurons to place in
%each layer.
%maxNetworkIterations: Maximim number of times the network will run before
%stopping, if not stopped by increase in error on the validation set
%earlier.
%theta: 3d parameter matrix. Use nan if training a new network. Use a
%previously obtained value in order to improve upon that value.
%improveTheta: boolean. If theta has a value and improveTheta is 1, the
%input value of theta will be improved upon.
%OUTPUT:
%[validationWeights, weights, validationErrors]

[trainWidth, trainHeight, numTrainingSamples]= size(trainData);
[validationWidth, validationHeight, numValidationSamples]=size(validationData);
validationErrors=[];

L = length(nil); %Number of layers. Note that there are L-1 layers in theta

%Training constant information
alpha = 1;

if isnan(theta) && ~improveTheta
    %Initialize random parameters -.5 to .5
    for n = 1:length(nil)-1
        theta(1:nil(n), 1:nil(n+1), n) = rand(nil(n), nil(n+1)) -0.5; %#ok<AGROW>
    end
end

%Initialize activations
activations(1:length(nil), 1:max(nil)) = 0;

%Initialize activation error matrix (2D)
deltaLC = zeros(size(activations,2), size(activations,1));

%Sigmoid hypothesis
h=@(x) 1./(1+exp(-x));

for iters=1:maxNetworkIterations
    
    %Initialize weight change matrix (3D). Must be done each iteration.
    deltaUC = zeros(size(theta,1), size(theta,2), size(theta,3));
    
    %Reset information
    error=0;
    display(iters); %Should be left in actual run code in order to display progress. This can be SLOW (days or weeks)
    for m=1:numTrainingSamples
        %Forward Propagation. This step is a factor of 1-3 faster than backprop.
        activations(1, 1:nil(1)) = reshape(trainData(:, :, m), trainWidth.*trainHeight, 1); %Unroll data;
        for layer = 2:L
            activations(layer, 1:nil(layer)) =  h( activations(layer-1, 1:nil(layer-1))*theta(1:nil(layer-1), 1:nil(layer), layer-1) );
        end
        
        %Backpropagation: Find theta updates
        %Output layer is easy, but must be done separately
        outputLCVec=activations(L, 1:nil(L))' - (trainLabels(m, 1:nil(L)))';
        deltaLC(1:nil(L), L) = outputLCVec;
        deltaUC(1:nil(L-1), 1:nil(L), L-1) = deltaUC(1:nil(L-1), 1:nil(L), L-1) + activations(L-1, 1:nil(L-1))'*deltaLC(1:nil(L),L)';
        
        %Harvest squared error in final layer
        error=error+sum(outputLCVec.^2);
        
        %Hidden layers
        for layer=L-1:-1:2 %No error in the input layer
            activationsInPreviousLayer=activations(layer, 1:nil(layer));
            
            %Application of analytical partial error derivative with respect to the weights.
            deltaLC(1:nil(layer),layer) = (theta(1:nil(layer), 1:nil(layer+1), layer)*deltaLC(1:nil(layer+1),layer+1))'.*activationsInPreviousLayer.*(1-activationsInPreviousLayer);
            
            %3D matrix subtraction to update all parameters
            deltaUC(1:nil(layer-1), 1:nil(layer), layer-1) = deltaUC(1:nil(layer-1), 1:nil(layer), layer-1) + activations(layer-1, 1:nil(layer-1))'*deltaLC(1:nil(layer),layer)';
        end
    end
    
    
    for layer=1:L-1
        tempDeltaUCLayer = deltaUC(1:nil(layer), 1:nil(layer+1), layer);
        deltaUC(1:nil(layer), 1:nil(layer+1), layer) = tempDeltaUCLayer./max(max(abs(tempDeltaUCLayer)));
    end
    
    
    %Update to theta. Performed in a manner such that error always goes
    %down (error minimum is never overshot).
    done=0;
    while ~done
        newTheta = theta - alpha.*deltaUC; %Theta update
        newError=0;
        for m=1:numTrainingSamples %Calculate new error
            %Forward Propagation. No need to store activation matrix.
            tempActivationVec = reshape(trainData(:, :, m), 1, trainHeight.*trainWidth); %Unroll data;
            for layer = 2:L
                tempActivationVec =  h(tempActivationVec*newTheta(1:nil(layer-1), 1:nil(layer), layer-1));
            end
            
            %Backpropagation for first layer only (find errors)
            newError=newError + sum((tempActivationVec - (trainLabels(m, :))).^2); %check order;
        end
        
        %Implementation of momentum for more efficient learning
        if newError<error
            theta=newTheta;
            alpha=alpha.*2;
            error=newError;
            done=1;
            display(alpha);
        else %Error has increased
            alpha=alpha./2;
        end
    end
    
    %Implementation of validation for early stopping to avoid overfitting
    validationErr=0;
    for m=1:numValidationSamples %Calculate new error
        %Forward Propagation. No need to store activation matrix.
        tempActivationVec = reshape(validationData(:, :, m), 1, validationHeight.*validationWidth); %Unroll data;
        for layer = 2:L
            tempActivationVec =  h(tempActivationVec*theta(1:nil(layer-1), 1:nil(layer), layer-1));
        end
        
        %Backpropagation for first layer only (find errors)
        validationErr=validationErr + sum((tempActivationVec - (validationLabels(m, :))).^2);
    end
    
    %Validation errors are useful for plotting purposes as well as for
    %stopping and should be saved.
    
    %Not preallocating here does little harm and helps later syntax
    validationErrors=[validationErrors, validationErr];
    
    %Choose the set of weights (theta) that better generalizes
    if length(validationErrors)>1 && validationErrors(end)<validationErrors(end-1)
        validationWeights=theta;
    end
    
    if length(validationErrors)>15 && max(validationErrors(end-15:end))==validationErrors(end)
        %Has been overfitting. Stop to save time.
        break;
    end
    
end

weights=newTheta;

end