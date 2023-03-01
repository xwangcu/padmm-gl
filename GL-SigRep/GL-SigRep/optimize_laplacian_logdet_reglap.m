function [W,sigmasquare] = optimize_laplacian_logdet_reglap(Y,lambda)

%% Laplacian constraints
[N,m] = size(Y);
%% optimization
cvx_begin

% cvx_solver mosek

variable Delta(N,N) semidefinite
variable W(N,N) symmetric
variable sigmasquare

minimize trace(1/m*Delta*Y*Y') - log_det(Delta) + 1/m*lambda*norm(reshape(W,N^2,1),1)

subject to
    Delta == diag(sum(W,2)) - W + eye(N)*sigmasquare % sigmasquare is the inverse of sigma^2 in the paper
    diag(W) == 0
    reshape(W,N^2,1) >= 0
    sigmasquare >= 10^(-4) % because strict inequality is discouraged

cvx_end