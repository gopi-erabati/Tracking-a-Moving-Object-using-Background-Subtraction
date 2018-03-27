%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUAL TRACKING
% ----------------------
% Background Subtraction on Car Sequence
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
imPath = 'car'; imExt = 'jpg';

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
    ImSeq(:,:,i) = imread(imgname); % load image
end
disp(' ... OK!');


% BACKGROUND SUBTRACTION
%=======================
%% 2.1 Frame differencing

% 2.1.1 difference of frames from one background model of all frames

bgModel = median(ImSeq, 3); %get the background model
hFig = figure(1);
SE = strel('square',3); % structring element for morphological opening!
for nFrame = 1:NumImages
    
    foreground = abs(ImSeq(:,:,nFrame) - bgModel) > 20; % sub each frames from bg
    foreground = imopen(foreground,SE);
    foreground = imdilate(foreground, SE);
    
    %     for bounding box using region props
    foregroundCC = bwconncomp(foreground);
    bbox = regionprops(foregroundCC, 'BoundingBox', 'Area');
    idx = find([bbox.Area] > 500);
    thisBB = bbox(idx).BoundingBox;
    
    %     %uncomment the code to get result of a frame in subplots
    %     if nFrame == 20
    %
    %         hFig = figure(6);
    %         subplot(2,2,1), imshow(ImSeq(:,:,20), []);  title('original image');
    %         subplot(2,2,2), imshow(bgModel, []); title('background model');
    %         subplot(2,2,3), imshow(foreground, []); title('foreground image');
    %         subplot(2,2,4), imshow(ImSeq(:,:,20), []); title('tracked vehicle');
    %         rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
    %             'EdgeColor','r','LineWidth',2 );
    % %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
    %     end
    
    hFig = figure(1);
    subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []);  title('original image');
    subplot(2,2,2), imshow(bgModel, []); title('background model');
    subplot(2,2,3), imshow(foreground, []); title('foreground image');
    subplot(2,2,4), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
    rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
        'EdgeColor','r','LineWidth',2 ); pause(0.000001);
    
end

%% 2.1.2 difference of frames from updated background model
figure(2);
title('bg model : frame differencing with updated bg model');
minFrames = 20;
SE = strel('square',2); % structring element for morphological opening!
for nFrame= minFrames:NumImages
    
    bgModel = median(ImSeq(:,:,1:nFrame-1), 3); %get the updated bg model
    foreground = abs(ImSeq(:,:,nFrame) - bgModel)> 40 ; % sub each frame from bg
    
    %     %morpholgical operations to remove white regions other than vehicle
    foreground = imopen(foreground,SE);
    %     foreground = imdilate(foreground, SE);
    
    %     for bounding box using region props
    foregroundCC = bwconncomp(foreground);
    bbox = regionprops(foregroundCC, 'BoundingBox', 'Area');
    idx = find([bbox.Area] > 500);
    thisBB = bbox(idx).BoundingBox;
    
    %         %uncomment the code to get result of a frame in subplots
    %     if nFrame == 40
    %
    %         hFig = figure(6);
    %         subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    %         subplot(2,2,2), imshow(bgModel, []); title('background model');
    %         subplot(2,2,3), imshow(foreground, []); title('foreground image');
    %         subplot(2,2,4), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
    %         rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
    %             'EdgeColor','r','LineWidth',2 );
    % %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
    %     end
    
    % show results
    hFig = figure(2);
    subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(2,2,2), imshow(bgModel, []); title('background model');
    subplot(2,2,3), imshow(foreground, []); title('foreground image');
    subplot(2,2,4), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
    rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
        'EdgeColor','r','LineWidth',2 ); pause(0.000001);
end

%% 2.1.3 difff of frames from updated background model with learning rate
figure(3);
title('bg model : frame differencing with updated bg model with learning rate');
minFrames = 25;
SE = strel('square', 2); % structring element for morphological opening!
for nFrame = minFrames:NumImages
    
    bgModel(:,:,nFrame-(minFrames-1)) = median(ImSeq(:,:,1:nFrame-1), 3); %background model for 'n-1' frames
    alpha = 0.05; %learning rate
    if nFrame > minFrames % from 'n+1' frames
        foreground = abs(ImSeq(:,:,nFrame) - bgModel(:,:,nFrame-(minFrames-1)))> 80 ; %to know whether pixel belongs to foreground or background
        %loop for checking pixels
        for nRow = 1:VIDEO_HEIGHT
            for nCol = 1:VIDEO_WIDTH
                if (foreground(nRow, nCol) == 255) %for pixels of foreground update the background pixels
                    bgModel(nRow, nCol, nFrame - (minFrames-1)) = alpha * ImSeq(:,:,nFrame) + (1 - alpha)* bgModel(nRow, nCol, nFrame - (minFrames));
                end
            end
        end
        foreground = abs(ImSeq(:,:,nFrame) - bgModel(:,:,nFrame-(minFrames-1)))> 40 ;
        
        %morphological operations
        foreground = imopen(foreground, SE);
        foreground = imdilate(foreground, SE);
        
        %     for bounding box using region props
        foregroundCC = bwconncomp(foreground);
        bbox = regionprops(foregroundCC, 'BoundingBox', 'Area');
        idx = find([bbox.Area] > 500);
        thisBB = bbox(idx).BoundingBox;
        
        %         %         uncomment the code to get result of a frame in subplots
        %         if nFrame == 35
        %
        %             hFig = figure(6);
        %             subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
        %             subplot(2,2,2), imshow(bgModel(:,:,nFrame-(minFrames-1)), []); title('background model');
        %             subplot(2,2,3), imshow(foreground, []); title('foreground image');
        %             subplot(2,2,4), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
        %             rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
        %                 'EdgeColor','r','LineWidth',2 );
        %             %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
        %         end
        
        % show results
        figure(3);
        subplot(2,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
        subplot(2,2,2), imshow(bgModel(:,:,nFrame-(minFrames-1)), []); title('background model');
        subplot(2,2,3), imshow(foreground, []); title('foreground image');
        subplot(2,2,4), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
        rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
            'EdgeColor','r','LineWidth',2 ); pause(0.00000001);
    end
end


%% 2.2 Gaussian Model

%intialise the mean and variance values
muPrevious = ImSeq(:,:,1);
variancePrevious = 800 * ones(VIDEO_HEIGHT, VIDEO_WIDTH);

%initialise the result
result = zeros(VIDEO_HEIGHT, VIDEO_WIDTH);
alpha = 0.01;
figure(4);
title('bg model : running avg guassian');
SE = strel('square', 2);
for nFrame = 1:NumImages
    
    %update the mean and variance
    mu = alpha * ImSeq(:,:,nFrame) + (1- alpha) * muPrevious;
    d = abs(ImSeq(:,:,nFrame) - mu);
    variance = alpha * d.^2 + (1-alpha)*variancePrevious;
    
    %check bg and fg
    value = abs(ImSeq(:,:,nFrame) - mu);
    result = value > 2.5*sqrt(variance);
    
    muPrevious = mu;
    variancePrevious = variance;
    
    %morphological opertaions
    result = imopen(result, SE);
    result = imdilate(result, SE);
    
    %     for bounding box using region props
    %     foregroundCC = bwconncomp(result);
    %     bbox = regionprops(result, 'BoundingBox', 'Area');
    %     idx = find([bbox.Area] > 500);
    %     thisBB = bbox(idx).BoundingBox;
    
    %     %         uncomment the code to get result of a frame in subplots
    %     if nFrame == 35
    %
    %         hFig = figure(6);
    %         subplot(1,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    %         subplot(1,2,2), imshow(result, []); title('foreground image');
    %     end
    
    %     show results
    figure(4);
    subplot(1,2,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(1,2,2), imshow(result, []); title('foreground image');    pause(0.00000001); 
    %     rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
    %         'EdgeColor','r','LineWidth',2 ); pause(0.00000001);
end


%% 2.3 EIGEN Model

figure(5);
title('bg model : Eigen');
% take 20 frames to form eigen model
imVec = [];
for nFrame = 1:20
    
    imVec = [imVec reshape(ImSeq(:,:,nFrame),[numel(ImSeq(:,:,nFrame)),1])];
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
singVal  = diag(S);
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
%project all other images onto eigenBG
for nFrame = 21:NumImages
    
    imgVec = reshape(ImSeq(:,:,nFrame),[numel(ImSeq(:,:,nFrame)),1]);
    imgProj = eigenBG' * (imgVec - meanImage);
    imgProj = eigenBG * imgProj + meanImage;
    
    %threshold the difference to detect moving objects
    result = abs(imgProj - imgVec) > 35;
    
    %reshape image
    result = reshape(result, [240 320]);
    
    %morphology
    result = imopen(result, SE);
    result = imdilate(result, SE);
    
    %     for bounding box using region props
    foregroundCC = bwconncomp(result);
    bbox = regionprops(result, 'BoundingBox', 'Area');
    idx = find([bbox.Area] > 500);
    thisBB = bbox(idx).BoundingBox;
    
%     %         uncomment the code to get result of a frame in subplots
%     if nFrame == 45
%         
%         hFig = figure(6);
%         subplot(1,3,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
%         subplot(1,3,2), imshow(result, []);  title('foreground image');
%         subplot(1,3,3), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
%         rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
%             'EdgeColor','r','LineWidth',2 );
%         %         title(hAxes, 'bg model : frame differencing with one bg model from all frames');
%     end
    
    %     show results
    figure(5);
    subplot(1,3,1), imshow(ImSeq(:,:,nFrame), []); title('original image');
    subplot(1,3,2), imshow(result, []);  title('foreground image');
    subplot(1,3,3), imshow(ImSeq(:,:,nFrame), []); title('tracked vehicle');
    rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
        'EdgeColor','r','LineWidth',2 ); pause(0.00000001);
    
end






