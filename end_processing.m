function [resbest, onetime] = end_processing(C,Y) % post-processing-2
maxIter = 100;
C = C ./ repmat(sqrt(sum(C.^2, 2)), 1, size(C,2));
times = zeros(maxIter,1);
for iter = 1:maxIter
    tic;            
    temp = litekmeans(C,length(unique(Y)),'MaxIter',100, 'Replicates',20);        
    pred = temp(:);
    times(iter) = toc;
    measurements(iter,:) = Clustering8Measure(Y,pred);
    %fprintf('ACC:%.6f, NMI:%.6f, Purity:%.6f, Fscore:%.6f, Precision:%.6f, Recall:%.6f, AR:%.6f\n',measurements(iter,1:7));
end
resbest = max(measurements,[],1);
onetime = mean(times);