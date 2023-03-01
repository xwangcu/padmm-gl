function L = wadjv2cgl(w)
%This function convert a edge weights vector w \in R^{p(p-1)/2} 
%     to the combinatorial graph Laplacian L \in R^{p \times p}.
W = squareform_sp(w);
D = diag(sum(W));
L = D-W;
end