%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUAL TRACKING
% ----------------------
% Background Subtraction on Highway sequence
% ----------------
% Date: October 2017
% Author: Gopikrishna Erabati
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

%%%%% LOAD THE IMAGES
%=======================
% Give image directory and extension
imPath = 'highway\input'; imExt = 'jpg';

% check if directory and files exist
if isdir(imPath) == 0
    error('USER ERROR : The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');
% Get image parameters
imgname = [imPath filesep filearray(1).name]; % get image name
I = imread(imgname); % read the 1st image and pick its size
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for i=1:NumImages
    imgname = [imPath filesep filearray(i).name]; % get image name
    ImSeq(:,:,i) = rgb2gray(imread(imgname)); % load image
end
disp(' ... OK!');


%%get ground truth images
% Give image directory and extension
imPath = 'highway\groundtruth'; imExt = 'png';

% check if directory and files exist
if isdir(imPath) == 0
    error('USER ERROR : The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading ground truth image files from the video sequence, please be patient...');
% Get image parameters
imgname = [imPath filesep filearray(1).name]; % get image name
I = imread(imgname); % read the 1st image and pick its size
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

gtSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for i=1:NumImages
    imgname = [imPath filesep filearray(i).name]; % get image name
    gtSeq(:,:,i) = imread(imgname); % load image ground truth
end
disp(' ... OK!');

% BACKGROUND SUBTRACTION
%=======================
%% 2.1 Frame differencing

% 2.1.1 difference of frames from one background model of all frames

bgModel = median(ImSeq, 3); %get the background model
hFig = figure(1);
SE = strel('square',2); % structring element for morphological opening!
TP = 0; TN = 0; FP = 0; FN = 0;
for nFrame = 471:NumImages

    foreground = abs(ImSeq(:,:,nFrame) - bgModel) > 20; % sub each frames from bg

    %morph op
    foreground = imerode(foreground, SE);
    foreground = imopen(foreground, SE);
    foreground = imfill(foreground, 'holes');

%     %uncomment the code to get result of a frame in subplots
%         if nFrame == 680
% 
%             hFig = figure(6);
%             subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
%             subplot(2,2,2), imshow(bgModel, []); title('background model');
%             subplot(2,2,3), imshow(foreground, []); title('foreground image')
%             subplot(2,2,4), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
%     %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
%         end
%         nFrame


    hFig = figure(1);
    subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(2,2,2), imshow(bgModel, []); title('background model');
    subplot(2,2,3), imshow(foreground, []); title('foreground image')
    subplot(2,2,4), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
    hold on; pause(0.0000000001);

    %calculation of precsion and recall
    TP = TP + nnz(foreground & gtSeq(:,:,nFrame));
    TN = TN + nnz(~foreground & ~gtSeq(:,:,nFrame));
    FP = FP + nnz(foreground & ~gtSeq(:,:,nFrame));
    FN = FN + nnz(~foreground & gtSeq(:,:,nFrame));
end
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('The precsion, recall and F-score of bg model : frame differencing with one bg model from all frames ...');
precision = TP / (TP + FP);
recall = TP / (TP +  FN);
FScore = 2*((precision * recall) / (precision + recall));
disp(['precision : ', num2str(precision)]);
disp(['recall : ', num2str(recall)]);
disp(['F Score : ', num2str(FScore)]);
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

%% 2.1.2 difference of frames from updated background model
figure(2);
minFrames = 471;
SE = strel('square',2); % structring element for morphological opening!
TP = 0; TN = 0; FP = 0; FN = 0;
for nFrame= minFrames:NumImages

    bgModel = median(ImSeq(:,:,1:nFrame-1), 3); %get the updated bg model
    foreground = abs(ImSeq(:,:,nFrame) - bgModel) > 20 ; % sub each frame from bg

    %     %morpholgical operations to remove white regions other than vehicle
    foreground = imerode(foreground, SE);
    foreground = imopen(foreground, SE);
    foreground = imfill(foreground, 'holes');

%     %uncomment the code to get result of a frame in subplots
%         if nFrame == 680
% 
%             hFig = figure(6);
%             subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
%             subplot(2,2,2), imshow(bgModel, []); title('background model');
%             subplot(2,2,3), imshow(foreground, []); title('foreground image')
%             subplot(2,2,4), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
%     %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
%         end
%         nFrame


    hFig = figure(2);
    subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(2,2,2), imshow(bgModel, []); title('background model');
    subplot(2,2,3), imshow(foreground, []); title('foreground image')
    subplot(2,2,4), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
    hold on; pause(0.0000000001);

    %calculation of precsion and recall
    TP = TP + nnz(foreground & gtSeq(:,:,nFrame));
    TN = TN + nnz(~foreground & ~gtSeq(:,:,nFrame));
    FP = FP + nnz(foreground & ~gtSeq(:,:,nFrame));
    FN = FN + nnz(~foreground & gtSeq(:,:,nFrame));
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('The precsion, recall and F-score of bg model : frame differencing with updated bg model ...');
precision = TP / (TP + FP);
recall = TP / (TP +  FN);
FScore = 2*((precision * recall) / (precision + recall));
disp(['precision : ', num2str(precision)]);
disp(['recall : ', num2str(recall)]);
disp(['F Score : ', num2str(FScore)]);
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');


%% 2.1.3 difff of frames from updated background model with learning rate
figure(3);
minFrames = 471;
SE = strel('square', 2); % structring element for morphological opening!
TP = 0; TN = 0; FP = 0; FN = 0;
for nFrame = minFrames:NumImages

    bgModel(:,:,nFrame-(minFrames-1)) = median(ImSeq(:,:,1:nFrame-1), 3); %background model for 'n-1' frames
    alpha = 0.05; %learning rate
    if nFrame > minFrames % from 'n+1' frames
        foreground = abs(ImSeq(:,:,nFrame) - bgModel(:,:,nFrame-(minFrames-1)))> 20 ; %to know whether pixel belongs to foreground or background
        %loop for checking pixels
        for nRow = 1:VIDEO_HEIGHT
            for nCol = 1:VIDEO_WIDTH
                if (foreground(nRow, nCol) == 255) %for pixels of foreground update the background pixels
                    bgModel(nRow, nCol, nFrame - (minFrames-1)) = alpha * ImSeq(:,:,nFrame) + (1 - alpha)* bgModel(nRow, nCol, nFrame - (minFrames));
                end
            end
        end
        foreground = abs(ImSeq(:,:,nFrame) - bgModel(:,:,nFrame-(minFrames-1)))> 20 ;

        %morphological operations
        foreground = imerode(foreground, SE);
        foreground = imopen(foreground, SE);
        %         foreground = imdilate(foreground, SE);
        foreground = imfill(foreground, 'holes');

%     %uncomment the code to get result of a frame in subplots
%         if nFrame == 680
% 
%             hFig = figure(6);
%             subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
%             subplot(2,2,2), imshow(bgModel(:,:,nFrame-(minFrames-1)), []); title('background model');
%             subplot(2,2,3), imshow(foreground, []); title('foreground image')
%             subplot(2,2,4), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
%     %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
%         end
%         nFrame


    hFig = figure(3);
    subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(2,2,2), imshow(bgModel(:,:,nFrame-(minFrames-1)), []); title('background model');
    subplot(2,2,3), imshow(foreground, []); title('foreground image')
    subplot(2,2,4), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
    hold on; pause(0.0000000001);

        %calculation of precsion and recall
        TP = TP + nnz(foreground & gtSeq(:,:,nFrame));
        TN = TN + nnz(~foreground & ~gtSeq(:,:,nFrame));
        FP = FP + nnz(foreground & ~gtSeq(:,:,nFrame));
        FN = FN + nnz(~foreground & gtSeq(:,:,nFrame));
    end
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('The precsion, recall and F-score of frame differencing with updated bg model with learning rate ...');
precision = TP / (TP + FP);
recall = TP / (TP +  FN);
FScore = 2*((precision * recall) / (precision + recall));
disp(['precision : ', num2str(precision)]);
disp(['recall : ', num2str(recall)]);
disp(['F Score : ', num2str(FScore)]);
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');



%% 2.2 Gaussian Model

%intialise the mean and variance values
muPrevious = mean(ImSeq(:,:,1:470),3);
variancePrevious = 100 * ones(VIDEO_HEIGHT, VIDEO_WIDTH);

%initialise the result
result = zeros(VIDEO_HEIGHT, VIDEO_WIDTH);
alpha = 0.01;
figure(4);
SE = strel('square', 2);
TP = 0; TN = 0; FP = 0; FN = 0;
for nFrame = 471:NumImages

    %update the mean and variance
    mu = alpha * ImSeq(:,:,nFrame) + (1- alpha) * muPrevious;
    d = abs(ImSeq(:,:,nFrame) - mu);
    variance = alpha * d.^2 + (1-alpha)*variancePrevious;

    %check bg and fg
    value = abs(ImSeq(:,:,nFrame) - mu);
    result = value > 1.5*sqrt(variance);

    muPrevious = mu;
    variancePrevious = variance;

    %morphological opertaions
    result = imerode(result, SE);
    %     result = imopen(result, SE);
    result = imfill(result, 'holes');

%     %uncomment the code to get result of a frame in subplots
%     if nFrame == 680
% 
%         hFig = figure(6);
%         subplot(1,3,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
%         subplot(1,3,2), imshow(result, []); title('foreground image')
%         subplot(1,3,3), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
%         %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
%     end
%     nFrame


    hFig = figure(4);
    subplot(1,3,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(1,3,2), imshow(result, []); title('foreground image')
    subplot(1,3,3), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
    hold on; pause(0.0000000001);

    %calculation of precsion and recall
    TP = TP + nnz(result & gtSeq(:,:,nFrame));
    TN = TN + nnz(~result & ~gtSeq(:,:,nFrame));
    FP = FP + nnz(result & ~gtSeq(:,:,nFrame));
    FN = FN + nnz(~result & gtSeq(:,:,nFrame));
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('The precsion, recall and F-score of bg model : running avg guassian ...');
precision = TP / (TP + FP);
recall = TP / (TP +  FN);
FScore = 2*((precision * recall) / (precision + recall));
disp(['precision : ', num2str(precision)]);
disp(['recall : ', num2str(recall)]);
disp(['F Score : ', num2str(FScore)]);
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');



%% 2.3 EIGEN Model

figure(5);
title('bg model : Eigen');
% take 20 frames to form eigen model
imVec = zeros(numel(ImSeq(:,:,1)),470);
for nFrame = 1:470
    
    imVec(:,nFrame) = reshape(ImSeq(:,:,nFrame),[numel(ImSeq(:,:,nFrame)),1]);
end

%compute mean image
meanImage = mean(imVec, 2);

%compute mean normalised image vectors
meanNormImg = imVec - repmat(meanImage, 1, size(imVec, 2));

%compute svd of mean norm images
[U,S,V] = svd(meanNormImg, 'econ');

%choose k as in PCA
% eigen value of 'k' principal componnets/sum of eigen value of all
% components > 0.95
singVal = diag(S);
for i = 1 : size(S,1)
    pl(i) = sum(singVal(1:i,1))/sum(singVal);
    if sum(singVal(1:i,1))/sum(singVal) > 0.95
        k = i;
        break;
    else
        continue;
    end
end


%eigen background
eigenBG = U(:, 1:k);

SE = strel('square', 2); % structring element for morphology
TP = 0; TN = 0; FP = 0; FN = 0;
%project all other images onto eigenBG
for nFrame = 471:NumImages
    
    imgVec = reshape(ImSeq(:,:,nFrame),[numel(ImSeq(:,:,nFrame)),1]);
    imgProj = eigenBG' * (imgVec - meanImage);
    imgProj = eigenBG * imgProj + meanImage;
    
    %threshold the difference to detect moving objects
    result = abs(imgProj - imgVec) > 18;
    
    %reshape image
    result = reshape(result, [240 320]);
    
    %morphological opertaions
    result = imerode(result, SE);
    result = imfill(result, 'holes');
    
%     %uncomment the code to get result of a frame in subplots
%     if nFrame == 680
%         
%         hFig = figure(6);
%         subplot(1,3,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
%         subplot(1,3,2), imshow(result, []); title('foreground image')
%         subplot(1,3,3), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
%         %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
%     end
%     nFrame
    
    
    hFig = figure(5);
    subplot(1,3,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(1,3,2), imshow(result, []); title('foreground image')
    subplot(1,3,3), imshow(gtSeq(:,:,nFrame), []); title('ground truth');
    hold on; pause(0.0000000001);
    
    %calculation of precsion and recall
    TP = TP + nnz(result & gtSeq(:,:,nFrame));
    TN = TN + nnz(~result & ~gtSeq(:,:,nFrame));
    FP = FP + nnz(result & ~gtSeq(:,:,nFrame));
    FN = FN + nnz(~result & gtSeq(:,:,nFrame));
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('The precsion, recall and F-score of bg model : Eigen ...');
precision = TP / (TP + FP);
recall = TP / (TP +  FN);
FScore = 2*((precision * recall) / (precision + recall));
disp(['precision : ', num2str(precision)]);
disp(['recall : ', num2str(recall)]);
disp(['F Score : ', num2str(FScore)]);
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');







