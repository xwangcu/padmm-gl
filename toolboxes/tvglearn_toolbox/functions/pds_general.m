function [X, stat] = pds_general(X, V, ops, params)
%  This function solves the following problem using primal dual splitting
%  min f(x) + g(x) + h(Lx)

% Input:
%      --- Required parameters
%      X                : Target variable
%      V{d}             : Dual variables
%      ops.f_grad       : Function F(X) to calculate the gradient of f.
%      ops.g_prox       : Function prox_g(X, gamma) to calculate the proximity operator of g
%      ops.h_prox{d}    : Function prox_h{d}(X, gamma) to calculate the proximity operator of h
%      ops.op_L{d}      : Liniear transform: out = Lx
%      ops.op_Lt{d}     : Liniear transform: out = Ltx
%      ops.lips         : Lipschitz constant of f
%      ops.opnorm       : Operator norm of LtL
%    
%      Note that the prox functions used in this function must be this
%      form F(X, gamma) even if gamma is unnecessary.
%        
%      --- Additional parameters
%      params.stepsize  : Prameter which control step size
%      params.maxiter   : Max iteration
%      params.tol       : Tolerance for stopping criterion

%% Default parameters
if nargin < 4
    params = struct;
end
if not(isfield(params, 'verbosity')),   params.verbosity = 1;   end
if not(isfield(params, 'maxiter')),       params.maxiter = 1000;      end
if not(isfield(params, 'tol')),         params.tol = 1e-5;      end
if not(isfield(params, 'step_size')),   params.step_size = .5;      end     % from (0, 1)
%%

if ops.lips == 0
    gamma1 = params.step_size;
    gamma2 = 1/(ops.opnorm*gamma1);
else
    gamma1 = (2/ops.lips)*params.step_size;
    gamma2 = 1/(ops.opnorm*gamma1) - ops.lips/(2*ops.opnorm);
end

num_dual = length(V);
tic
for itr = 1:params.maxiter
    Xpre = X;
    Vpre = V;
    sum_op_v = 0;
    for dd = 1 : num_dual
        sum_op_v = sum_op_v + ops.op_Lt{dd}(V{dd});
    end
    X = ops.g_prox(X - gamma1*(ops.f_grad(X)+sum_op_v), gamma1);
    diff_X = 2*X-Xpre;
    sum_delta_V = 0;
    sum_Vpre = 0;
    for dd = 1 : num_dual
        V{dd} = V{dd} + gamma2*ops.op_L{dd}(diff_X);
        V{dd} = V{dd} - gamma2*ops.h_prox{dd}(V{dd}/gamma2, 1/gamma2);
        delta_V = (V{dd} - Vpre{dd}).^2;
        l2_Vpre = (Vpre{dd}).^2;
        sum_delta_V = sum_delta_V + sum(delta_V(:));
        sum_Vpre = sum_Vpre + sum(l2_Vpre(:));
    end
    
    % plot update amount
    update_amount_primal = norm(X - Xpre, 'Fro') / norm(Xpre, 'Fro');
    update_amount_dual = sqrt(sum_delta_V) / sqrt(sum_Vpre);
    if params.verbosity > 1
        fprintf('iter %4d:  primal %6.4e  dual  %6.3e \n', itr, update_amount_primal, update_amount_dual);
    end
    
    % stopping criterion
    if update_amount_primal < params.tol
        break
    end
    
end
stat.time = toc;
stat.V = V;
stat.itr = itr;
if params.verbosity > 0
    fprintf('iter %4d:  primal %6.4e  dual  %6.3e \n', itr, update_amount_primal, update_amount_dual);
    fprintf('Time needed is %f seconds\n', stat.time);
end
end