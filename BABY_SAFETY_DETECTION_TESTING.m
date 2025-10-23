% Load trained model (no retraining needed)
load babyNet.mat net

% Test with a new image
testImage = "C:\Users\Akshaa Karunya\Downloads\images-S.jpg";
img = imread(testImage);
imgResized = imresize(img, [224 224]);
label = classify(net, imgResized);

imshow(img);
title(['Predicted: ', char(label)]);
