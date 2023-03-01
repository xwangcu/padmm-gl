function [out] = calc_fmeasure(W, W_true)
tp = nnz(W>0 & W_true>0);
fp = nnz(W>0 & W_true==0);
fn = nnz(W==0 & W_true>0);
out = 2*tp/(2*tp+fn+fp);
end
