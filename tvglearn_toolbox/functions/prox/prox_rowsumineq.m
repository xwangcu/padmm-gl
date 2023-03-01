function[out] = prox_rowsumineq(X, d)
% This function return the prox of the equality constraint in colum sum.
% X: matrix
% d: scalar
X = X';
[x_row, ~] = size(X);
target_idx = sum(X)<d;
target_num = sum(target_idx);
d_vec = ones(1, target_num)*d;
X(:, target_idx) = X(:, target_idx) + 1/x_row*ones(x_row,1)*(d_vec - ones(1,x_row)* X(:, target_idx));
out = X';
end