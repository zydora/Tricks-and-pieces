function MNIST_class
% dataset download from 'https://cs.nyu.edu/~roweis/data.html'
load('C:\Users\46107\Desktop\Codes\0_Datasets\MNIST\raw\mnist_all.mat');
type = 'test';
savePath = 'C:\Users\46107\ALS\';
for num = 0:1:9
    numStr = num2str(num);
    tempNumPath = strcat(savePath, numStr);
    mkdir(tempNumPath);
    tempNumPath = strcat(tempNumPath,'\');
    tempName = [type, numStr];
    tempFile = eval(tempName);
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
