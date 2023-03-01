function [W, fval_cvx] = gl_cvx(X, alpha, beta)

% min_{W} trace(W*Z) - alpha*sum(log(sum(W0,2))) + 0.5*beta*(norm(W0,'fro'))^2
% s.t.    W>=0, W=W', diag(W)=0

%% initialization
DIM = size(X,1);
Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       Z(i,j) = norm(X(i,:)-X(j,:),2)^2; 
    end 
end

%% cvx

cvx_begin

cvx_precision best

variable W(DIM,DIM) symmetric

minimize trace(W*Z) - alpha*sum_log(sum(W,2)) + 0.5*beta*square_pos(norm(W,'fro'))

subject to
    W >= 0;
    diag(W) == 0;

cvx_end

fval_cvx = cvx_optval;

% % test
% fval_cvx_opt = trace(W*Z) - alpha*sum(log(sum(W,2))) + 0.5*beta*(norm(W,'fro'))^2;
% fprintf('\nfval_cvx_opt=%f-------------------------------------\n', fval_cvx_opt);