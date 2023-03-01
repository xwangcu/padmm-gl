function [re] = calc_relative_error(W, W_true)

if isvector(W)
    w = W;
    w_true = W_true;
else
    w = squareform_sp(W);
    w_true = squareform_sp(W_true);
end
w = rescale(w);
w_true = rescale(w_true);
re = norm(w - w_true, 'fro')/norm(w_true, 'fro');
end