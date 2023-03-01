function[out] = prox_rowsumeq(X, d)
% This function return the prox of the equality constraint in colum sum.
% X: matrix
% d: scalar
X = X';
[x_row, x_col] = size(X);
d_vec = ones(1, x_col)*d;
out = X + 1/x_row*ones(x_row,1)*(d_vec - ones(1,x_row)*X);
out = out';
end