function w = cgl2wadjv(L)
%This function convert a combinatorial graph Laplacian L \in R^{p \times p}
%     to the edge weights vector w \in R^{p(p-1)/2}.
d = diag(L);
num_nodes = length(d);
L = -L;
W = L + L' + repmat(d, [1,num_nodes]) + repmat(d', [num_nodes, 1]);
W(1:num_nodes+1:end) = 0;
w = transpose(squareform(W));
end