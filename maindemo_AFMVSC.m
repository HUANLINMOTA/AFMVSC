clc;
clear;
addpath(genpath('./'));
ds{1} = 'BBCSport';
%ds{1} = 'handwritten';
%ds{1} = 'uci-digit';
%ds{1} = 'ALOI_100';  
dataName = ds{1}; 
disp(dataName);
load(dataName);
%Y = truth;  % handwritten
%X = cell(3,1); X{1} = mfeat_fac; X{2} = mfeat_fou; X{3} = mfeat_kar; Y = truth;  % uci-digit
X = cell(2,1); X{1} = sport01; X{2} = sport02; Y = truth;  % BBCSport
numsample = length(Y);
k = length(unique(Y));
L = cell(length(X),1);
numview = length(X);
L_sum = zeros(numsample,numsample);
for i = 1:numview 
   X{i} = double(X{i});    
   X{i} = X{i}';   % if need
   X{i} = X{i} ./ repmat(sqrt(sum(X{i} .^ 2,1)), size(X{i},1),1); 
   L{i} = Laplacians(X{i}',5);  % 2~8 
   L_sum = L_sum + L{i};
end       
lambda1 = 17; % 13, 17, 18, 19, 50 or others
lambda2 = 1.1; % 1.1, 1.6, 1.7, 1.8 or others 
lambda3 = 14; % 12, 14 or others 
Q = inv(L_sum + lambda2 * numview * eye(length(L_sum)));
tic;
[U,Z,ob] = AFMVSC(X,L,lambda1,lambda2,lambda3,k,numsample,Q); 
time1  = toc;
[res, time2] = end_processing(U,Y);
fprintf('Time:%.6f\n', time1 + time2);
fprintf('ACC:%.6f, NMI:%.6f, Purity:%.6f, Fscore:%.6f, Precision:%.6f, Recall:%.6f, AR:%.6f\n',res(1:7));