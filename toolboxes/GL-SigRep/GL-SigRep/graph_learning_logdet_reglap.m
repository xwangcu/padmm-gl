function [Loutput,sigmasquare] = graph_learning_logdet_reglap(X,param)
% Learning sparse graph Laplacians (Lake et al.)
% Signals X follow Gaussian assumption (each column is a signal)

[W,sigmasquare] = optimize_laplacian_logdet_reglap(X,param.lambda);
Loutput = sgwt_laplacian(W,'opt','raw');
% Loutput = Loutput/trace(Loutput)*param.N;

if sigmasquare <= 0
 error('sigma is not positive :(')
end