function [result,Z,ob] = AFMVSC(X,L,lambda1,lambda2,lambda3,k,numsample,Q)
%% initialize
maxIter = 100 ;
numview = length(X);

W = cell(numview,1);
A = zeros(k,k);
C = zeros(k,numsample);
G = zeros(k,numsample);
for i = 1:numview   
   di = size(X{i},1); 
   W{i} = zeros(di,k);
end
C(:,1:k) = eye(k);
alpha = ones(1,numview) / numview;
%% optimize
flag = 1;
iter = 0;
while flag
    iter = iter + 1;
    
    % optimize Wi
    AZ = A * C;
    for iv=1:numview        
        D = X{iv} * AZ';              
        [UD,~,VD] = svd(D,'econ');
        W{iv} = UD * VD';
    end

    % optimize A
    Alphasum = 0;
    F = 0;
    for iv = 1:numview
        al2 = alpha(iv)^2;
        Alphasum = Alphasum + al2;
        F = F + al2 * W{iv}' * X{iv} * C';
    end
    [UF,~,VF] = svd(F,'econ');
    A = UF * VF';
  
    % optimize C
    H = 2 * Alphasum * eye(k) + 2* lambda1 * eye(k);
    H = (H + H') / 2;
    options = optimset('Algorithm','interior-point-convex','Display','off');
    for j=1:numsample
        ff = 0;
        for iv=1:numview
            ff = ff - 2 * alpha(iv)^2 * A' * W{iv}' * X{iv}(:,j);
        end
        ff = ff - lambda3 * G(:,j);
        C(:,j) = quadprog(H,ff,[],[],ones(1,k),1,zeros(k,1),ones(k,1),[],options);
    end

    % optimize G        
    P = lambda3 / 2 * C;
    G = max(0, P * Q);
    
    % optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm(X{iv} - W{iv} * A * C,'fro')^2;
    end
    Mfra = M .^ -1;
    J = 1 / sum(Mfra);
    alpha = J * Mfra;

    % convergence
    term1 = 0;
    term3 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A * C,'fro')^2;        
        term3 = term3 + trace(G * L{iv} * G');  
    end
    term2 = lambda1 * norm(C,'fro')^2;
    term4 = lambda2 * norm(G,'fro')^2;
    term5 = lambda3 * trace(C' * G);
    ob(iter) = term1 + term2 + term3 + term4 - term5;       
    %fprintf('Iter: %d, loss: %.11f.\n',iter,obj(iter));
    if (iter > 9) && (norm(ob(iter - 1) - ob(iter),'fro') < norm(ob(iter - 1),'fro') * 1e-3 || iter > maxIter)       
        flag = 0;
    end
end
%% post-processing-1
Z = (C .* G)';
[UZ,~,~]=svd(Z,'econ');
result = UZ;
end
         
         
    
