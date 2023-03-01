function [W_all, fval_pds, primal_gap_iter, stat] = tvglearn_tikhonov(Z, alpha, beta, eta, w_opt, params)

% Kalfolias et al. method [Kalofolias, et al. ICASSP2017]
% This code is implemented based on gsp_learn_graph_log_degrees (refer the following URL).
% https://epfl-lts2.github.io/gspbox-html/doc/learn_graph/gsp_learn_graph_log_degrees.html
%
%   Inputs:
%         Z         : Tenosr which have T matrix with (squared) pairwise
%                     distances of nodes. (T: time size)
%         alpha     : Log prior constant  (bigger a -> bigger weights in W)
%         beta      : W||_F^2 prior constant  (bigger b -> more dense W)
%         eta       : L1 norm ||W(t) - W(t-1)||_2^2 parameter
%         param     : Optional parameters
%
%   Outputs:
%         W         : Tenosr with T weighted adjacency matrix
%         stat      : Optional output statistics (adds small overhead)



%% Default parameters
if nargin < 5
    params = struct;
end

if not(isfield(params, 'verbosity')),   params.verbosity = 2;   end
if not(isfield(params, 'maxit')),       params.maxit = 1000;      end
if not(isfield(params, 'tol')),         params.tol = 1e-3;      end
if not(isfield(params, 'step_size')),   params.step_size = .5;      end     % from (0, 1)
if not(isfield(params, 'fix_zeros')),   params.fix_zeros = false;      end
if not(isfield(params, 'max_w')),       params.max_w = inf;         end

%% Fix parameter size and initialize

primal_gap_iter = zeros(params.maxit,1);

if ndims(Z) == 3
    [num_nodes, ~, T] = size(Z);
    num_edges = num_nodes*(num_nodes-1)/2;
    z_all = zeros(num_edges, T);
    for ii=1:T
        z_all(:,ii) = squareform_sp(Z(:,:,ii));
    end
else
    [num_edges, T] = size(Z);
    num_nodes = round((1+sqrt(1+8*num_edges))/2);
    z_all = Z;
end

if isfield(params, 'w_0')
    if not(isfield(params, 'c'))
        error('When params.w_0 is specified, params.c should also be specified');
    else
        c = params.c;
    end
    if isvector(params.w_0)
        w_0 = params.w_0;
    else
        w_0 = squareform_sp(params.w_0);
    end
    w_0 = w_0(:);
else
    w_0 = 0;
end

% if sparsity pattern is fixed we optimize with respect to a smaller number
% of variables, all included in w
if params.fix_zeros
    if not(isvector(params.edge_mask))
        params.edge_mask = squareform_sp(params.edge_mask);
    end
    % use only the non-zero elements to optimize
    ind = find(params.edge_mask(:));
    z_all = full(z_all(ind,:));
    if not(isscalar(w_0))
        w_0 = full(w_0(ind));
    end
else
    z_all = full(z_all);
    w_0 = full(w_0);
end

w_all = zeros(size(z_all));

%% Needed operators
% S*w = sum(W)
if params.fix_zeros
    [S, St] = sum_squareform(num_nodes, params.edge_mask(:));
else
    [S, St] = sum_squareform(num_nodes);
end

% S: edges -> nodes
K_op = @(w_all) S*w_all;

% S': nodes -> edges
Kt_op = @(z_all) St*z_all;

if params.fix_zeros
    norm_K = normest(S);
    % approximation: 
    % sqrt(2*(n-1)) * sqrt(nnz(params.edge_mask) / (n*(n+1)/2)) /sqrt(2)
else
    % the next is an upper bound if params.fix_zeros
    norm_K = sqrt(2*(num_nodes-1));
end

%% Learn the graph

% put proximal of trace plus positivity together
f.eval = @(w) sum(sum(2*w.*z_all));    % half should be counted
f.prox = @(w, c) min(params.max_w, max(0, w - 2*c*z_all));  % all change the same

param_prox_log.verbose = params.verbosity - 3;
g.eval = @(z) -alpha * sum(sum(log(z)));
g.prox = @(z, c) prox_sum_log(z, c*alpha, param_prox_log);
% proximal of conjugate of g: z-c*g.prox(z/c, 1/c)
g_star_prox = @(z, c) z - c * prox_sum_log(z/c, alpha/c, param_prox_log);

if w_0 == 0
    % "if" not needed, for c = 0 both are the same but gains speed
    h.eval = @(w) beta * sum(vecnorm(w).^2) + eta*(sum(vecnorm(w(:,2:end)-w(:,1:end-1)).^2));
    h.grad = @(w) 2*(beta + 2*eta)*w - eta*([w(:,1) w(:,1:end-1)] + [w(:,2:end) w(:,end)]);
    h.beta = 2 * beta;
else
    w_0 = repmat(w_0, [1 T]);
    h.eval = @(w) beta * sum(vecnorm(w).^2) + c*sum(vecnorm(w - w_0).^2) + eta*(sum(vecnorm(w(:,2:end)-w(:,1:end-1)).^2));
    h.grad = @(w) 2*w*(beta + c + 2*eta) - 2*c*w_0 - eta*([w(:,1) w(:,1:end-1)] + [w(:,2:end) w(:,end)]);    %2 * ((beta+c) * w - c * w_0);
    h.beta = 2*(beta + c + 2*eta);
end

%% My custom FBF based primal dual (see [1] = [Komodakis, Pesquet])
% parameters mu, epsilon for convergence (see [1])
mu = h.beta + norm_K;     %TODO: is it squared or not??
epsilon = lin_map(0.0, [0, 1/(1+mu)], [0,1]);   % in (0, 1/(1+mu) )

% INITIALIZATION
% primal variable ALREADY INITIALIZED
%w = params.w_init;
% dual variable
v_n = K_op(w_all);  %d_i

if nargout > 1 || params.verbosity > 1
    stat.f_eval = nan(params.maxit, 1);
    stat.g_eval = nan(params.maxit, 1);
    stat.h_eval = nan(params.maxit, 1);
    stat.fgh_eval = nan(params.maxit, 1);
    stat.pos_violation = nan(params.maxit, 1);
end
if params.verbosity > 1
    fprintf('Relative change of primal, dual variables, and objective fun\n');
end

tic
gn = lin_map(params.step_size, [epsilon, (1-epsilon)/mu], [0,1]); % in [epsilon, (1-epsilon)/mu]

for i = 1:params.maxit

    % added by wxl
    primal_gap_iter(i) = norm(w_all(:)-w_opt); % commented when comparing runtime
    
    Y_n = w_all - gn * (h.grad(w_all) + Kt_op(v_n));  %y_i
    y_n = v_n + gn * (K_op(w_all));  % \bar{y_i}
    
    P_n = f.prox(Y_n, gn); %p_i
    p_n = g_star_prox(y_n, gn); % = y_n - gn*g_prox(y_n/gn, 1/gn)   % \bar{p_i}
    Q_n = P_n - gn * (h.grad(P_n) + Kt_op(p_n));  % q_i
    q_n = p_n + gn * (K_op(P_n));   % \bar{q_i}
    
    if nargout > 1 || params.verbosity > 2
        stat.f_eval(i) = f.eval(w_all);
        stat.g_eval(i) = g.eval(K_op(w_all));
        stat.h_eval(i) = h.eval(w_all);
        stat.fgh_eval(i) = stat.f_eval(i) + stat.g_eval(i) + stat.h_eval(i);
        stat.pos_violation(i) = -sum(min(0,w_all(:)));
    end
    rel_norm_primal = sum(vecnorm(- Y_n + Q_n))/sum(vecnorm(w_all));
    rel_norm_dual = sum(vecnorm(- y_n + q_n))/sum(vecnorm(v_n));
    
    if params.verbosity > 3
        fprintf('iter %4d: %6.4e   %6.4e   %6.3e \n', i, rel_norm_primal, rel_norm_dual, stat.fgh_eval(i));
    elseif params.verbosity > 2
        fprintf('iter %4d: %6.4e   %6.4e   %6.3e\n', i, rel_norm_primal, rel_norm_dual, stat.fgh_eval(i));
    elseif params.verbosity > 1
        fprintf('iter %4d: %6.4e   %6.4e\n', i, rel_norm_primal, rel_norm_dual);
    end
    
    w_all = w_all - Y_n + Q_n;
    v_n = v_n - y_n + q_n;
    
    if rel_norm_primal < params.tol && rel_norm_dual < params.tol 
        break
    end
end

fval_pds = stat.fgh_eval(min(i,params.maxit));

stat.time = toc;
if params.verbosity > 0
    fprintf('# iters: %4d. Rel primal: %6.4e Rel dual: %6.4e  OBJ %6.3e\n', i, rel_norm_primal, rel_norm_dual, f.eval(w_all) + g.eval(K_op(w_all)) + h.eval(w_all));
    fprintf('Time needed is %f seconds\n', stat.time);
end

%%

if params.verbosity > 3
    figure; plot(real([stat.f_eval, stat.g_eval, stat.h_eval])); hold all; plot(real(stat.fgh_eval), '.'); legend('f', 'g', 'h', 'f+g+h');
    figure; plot(stat.pos_violation); title('sum of negative (invalid) values per iteration')
    figure; semilogy(max(0,-diff(real(stat.fgh_eval'))),'b.-'); hold on; semilogy(max(0,diff(real(stat.fgh_eval'))),'ro-'); title('|f(i)-f(i-1)|'); legend('going down','going up');
end

if params.fix_zeros
    w_full = zeros(num_edges, T);
    w_full(ind, 1:end) = w_all(:, 1:end);
    w_all = w_full;
    %w = sparse(ind, ones(size(ind)), w, l, 1);
end

if ndims(Z) == 3
    W_all = zeros(num_nodes, num_nodes, T);
    for ii=1:T
        W_all(:,:,ii) = squareform_sp(w_all(:,ii));
    end
else
    W_all = w_all;
end

end
