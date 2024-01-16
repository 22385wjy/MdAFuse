function res = fusionAssess(im1, im2, fused)

close all; clear; clc
rootPath = '../../testResults/';
filename = [rootPath, '6metrics_74MR-SPECT.xlsx'];
title = {'EN', 'MI', 'SD', 'SCD', 'rSFe'};
xlswrite(filename, title, '74MR-SPECT');

mriPpetP = [rootPath, 'MR-SPECT_74/originalIms/'];
fileFolser = ls(mriPpetP);
test_num = length(fileFolser);
Ai = 0;
resultMatrix = [];  

for i = 1:test_num-2
    i2 = i-1;
    ImName = num2str(i2);
    fileP = [mriPpetP, ImName, '/'];
    originalList_ = dir(strcat(fileP, '*.png'));

    mriP = [fileP, 'mri_', num2str(i2), '.png'];
    petP = [fileP, 'spect_', num2str(i2), '.png'];
    Mri_im = imread(mriP);
    Pet_im = imread(petP);

    im1 = double(rgb2gray(Mri_im));
    im2 = double(rgb2gray(Pet_im));

    Ai = Ai + 1;
    lineN = ['A', num2str(Ai)];

    fusedPath = [rootPath, 'MR-SPECT_74/fusedIms', '/', char(ImName), '/'];
    fusedPath_list = dir(strcat(fusedPath, '*.png'));
    img_num = length(fusedPath_list);
    callQ = zeros(1, 3);
    if img_num > 0
        Q = zeros(img_num, 5);
        for Iim = 1:img_num
            image_name = fusedPath_list(Iim).name;
            fused = imread(strcat(fusedPath, image_name));
            if numel(size(fused)) > 2
                fused = rgb2gray(fused);
            end
            fused = imresize(fused, [256, 256]);
            image_fused = im2double(fused);
            Q(Iim, 1) = entropy(image_fused);
            Q(Iim, 2) = MI(fused, im1, im2);
            Q(Iim, 3) = analysis_sd(image_fused);
            Q(Iim, 4) = SCD(fused, im1, im2);
            Q(Iim, 5) = metricZheng(im1, im2, fused);
        end
        callQ = mean(Q, 1);  
    end

    Ai = Ai + 1;
    lineN = ['A', num2str(Ai)];
    xlswrite(filename, callQ, '74MR-SPECT', lineN);

    resultMatrix = [resultMatrix; callQ]; 
end

averageResult = mean(resultMatrix, 1);
averageResultCell = num2cell(averageResult);
[num, txt, raw] = xlsread(filename, '74MR-SPECT');
lastRow = size(raw, 1);


blankRow = cell(3, size(raw, 2));  
newData = [raw; blankRow; blankRow; blankRow; averageResultCell];  
xlswrite(filename, newData, '74MR-SPECT');

disp('EN  MI  SD SCD rSFe');
res = averageResult;  

end