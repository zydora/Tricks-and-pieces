% Generate class images of MNIST
function MNIST_class
% dataset download from 'https://cs.nyu.edu/~roweis/data.html'
load('mnist_all.mat');
type = 'train';
savePath = 'C:\Users\46107\ALS\';
for num = 0:1:9
    mkdir(strcat(savePath, num2str(num)));
    tempNumPath = strcat(tempNumPath,'\');
    tempFile = eval([type, numStr]);
    [height, ~]  = size(tempFile);
    for r = 1:1:height
        tempImg = reshape(tempFile(r,:),28,28)';
        tempImgPath = strcat(tempNumPath,num2str(r-1));
        tempImgPath = strcat(tempImgPath,'.bmp');
        imwrite(tempImg,tempImgPath);
    end
end
fprintf('MNIST_class Loaded Successfully \n');
end