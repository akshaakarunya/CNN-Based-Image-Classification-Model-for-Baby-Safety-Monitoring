imds = imageDatastore("C:\Users\Akshaa Karunya\OneDrive\Desktop\dataset", ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

tbl = countEachLabel(imds)   % to check labels and counts
% Split dataset into training (80%) and testing (20%)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Display how many images are in train & test
disp('Training set:');
countEachLabel(imdsTrain)

disp('Testing set:');
countEachLabel(imdsTest)

% Define image size for CNN (224x224x3 for RGB)
imageSize = [224 224 3];

% Preprocess with augmentation
augTrain = augmentedImageDatastore(imageSize, imdsTrain);
augTest  = augmentedImageDatastore(imageSize, imdsTest);
% Define a simple CNN for baby safety detection
layers = [
    imageInputLayer([224 224 3])   % input layer

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2)   % 2 classes: safe, danger
    softmaxLayer
    classificationLayer];

% Training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...   % increase to 20 or 30 for better results
    'InitialLearnRate',0.001, ...
    'Verbose',true, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(augTrain, layers, options);

% Test the network
YPred = classify(net, augTest);
YTest = imdsTest.Labels;

% Accuracy
accuracy = sum(YPred == YTest)/numel(YTest);
disp(['Test Accuracy: ', num2str(accuracy*100), '%']);
save babyNet.mat net
