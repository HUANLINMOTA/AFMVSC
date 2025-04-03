function L = Laplacians(X, k)
n = size(X, 1);
W = zeros(n, n);
for i = 1:n
    distances = sum((X - X(i, :)).^2, 2);
    [~, idx] = sort(distances);
    W(i, idx(2:k+1)) = 1; 
    W(idx(2:k+1), i) = 1; 
end
D = diag(sum(W, 2));
L = D-W; 
end