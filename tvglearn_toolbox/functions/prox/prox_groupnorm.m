function[out] = prox_groupnorm(x, T, gamma)
% This function output the proximal operator of group norm penalty.
%   Inputs:
%         x         : N*T vector; N and T are number of nodes and time
%                     slots respectively 
%         gamma     : parameter
%
X = reshape(x, [], T);
l2norm_group = sqrt(sum(X.*X));
X(:, l2norm_group <= gamma) = 0;
idx_non = l2norm_group > gamma;
X(:, idx_non) = X(:, idx_non) * diag(1 - gamma./l2norm_group(idx_non));
out = X(:);
end