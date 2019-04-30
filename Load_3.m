%% load dataset
clear all;clc
Dataset = 'MNIST';%You should change the W number nodes as same.
[TrainImages,TestImages,TrainLabels,TestLabels] = load_dataset_3(Dataset);