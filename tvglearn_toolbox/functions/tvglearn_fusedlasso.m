function [W_all, fval_pds, primal_gap_iter, stat] = tvglearn_fusedlasso(Z, alpha, beta, eta, w_opt, params)

% Estimate a dynamic graph from a spatial-time series signal.
% This code is written based on gsp_learn_graph_log_degrees (refer the following URL).
% https://epfl-lts2.github.io/gspbox-html/doc/learn_graph/gsp_learn_graph_log_degrees.html
%
%   Inputs:
%         Z         : Tenosr which have T matrix with (squared) pairwise
%                     distances of nodes. (T: time size)
%         alpha     : Log prior constant  (bigger a -> bigger weights in W)
%         beta      : W||_F^2 prior constant  (bigger b -> more dense W)
%         eta       : L1 norm |W(t) - W(t-1)| parameter
%         param     : Optional parameters
%
%   Outputs:
%         W         : Tenosr with T weighted adjacency matrix
%         stat      : Optional output statistics (adds small overhead)

%% Default parameters
if nargin < 5
    params = struct;
end

if not(isfield(params, 'verbosity')),   params.verbosity = 1;   end
if not(isfield(params, 'maxit')),       params.maxit = 1000;      end
if not(isfield(params, 'tol')),         params.tol = 1e-3;      end
if not(isfield(params, 'step_size')),   params.step_size = 0.9;      end     % from (0, 1), in otehr words, 0<step_size<1
if not(isfield(params, 'fix_zeros')),   params.fix_zeros = false;      end
if not(isfield(params, 'max_w')),       params.max_w = inf;         end

%% Fix parameter size and initialize

iter = params.maxit;
fval_iter = zeros(params.maxit,1);
primal_res_iter = zeros(params.maxit,1);
dual_res_iter = zeros(params.maxit,1);
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
num_edges = size(w_all,1);

%% Needed operators
% S*w = sum(W) 
if params.fix_zeros
    [S, St] = sum_squareform(num_nodes, params.edge_mask(:));   
else
    [S, St] = sum_squareform(num_nodes);
end

% D*w = w - w_pre
[D, Dt] = make_diff_matrix(num_edges, T);
S_multi = make_blkdiag_multi(S, T);

A = [D;S_multi];

% operator
S_op = @(w) reshape(S*reshape(w,[num_edges, T]), [], 1);
St_op = @(z) reshape(St*reshape(z,[num_nodes, T]), [], 1);
D_op = @(w) D*w;
Dt_op = @(z) Dt*z;

norm_K = normest(A, 1.e-4); % spectral norm = 2-norm
clearvars A S_multi

%% Learn the graph
w_vec = w_all(:);
z_vec = z_all(:);

% put proximal of trace plus positivity together
f.eval = @(w) 2*w.' * z_vec;    % half should be counted
f.prox = @(w, c) min(params.max_w, max(0, w - 2*c*z_vec));

param_prox_log.verbose = params.verbosity - 3;
%g.eval1 = @(v1, v2) -alpha*sum(log(v1)) + eta*norm(v2, 1) ;
g.eval1 = @(v1) -alpha*sum(log(v1));
g.eval2 = @(v2) eta*norm(v2, 1);

g.prox1 = @(z, c) prox_sum_log(z, c*alpha, param_prox_log);
g.prox2 = @(z, c) prox_l1(z, c*eta);
% proximal of conjugate of g: z-c*g.prox(z/c, 1/c)
g_star_prox1 = @(z, c) z - c * prox_sum_log(z/c, alpha/c, param_prox_log);
g_star_prox2 = @(z, c) z - c * prox_l1(z/c, eta/c);

if w_0 == 0
    % "if" not needed, for c = 0 both are the same but gains speed
    h.eval = @(w) beta * norm(w)^2;
    h.grad = @(w) 2 * beta * w;
    h.beta = 2 * beta;
else
    w_0 = reptmat(w_0, [T 1]);
    h.eval = @(w) beta * norm(w) + c*norm(w - w_0);
    h.grad = @(w) 2 * ((beta+c) * w - c * w_0);
    h.beta = 2 * (beta+c);
end

%% My custom FBF based primal dual (see [1] = [Komodakis, Pesquet])
% parameters mu, epsilon for convergence (see [1])
% mu = h.beta + norm_K;     %TODO: is it squared or not??
% epsilon = params.step_size * 1/(1+mu); % in (0, 1/(1+mu)) 
% gn = linspace(epsilon, (1-epsilon)/mu, params.maxit);  % in [epsilon, (1-epsilon)/mu]

% norm_K = sqrt(2*(num_node-1));
mu = h.beta + norm_K;     %TODO: is it squared or not??
epsilon = 1/(1+mu) * params.step_size;
gn = linspace(epsilon, (1-epsilon)/mu, params.maxit);
% epsilon = lin_map(0.0, [0, 1/(1+mu)], [0,1]);   % in (0, 1/(1+mu) )
% gn = lin_map(params.step_size, [epsilon, (1-epsilon)/mu], [0,1]); 
%gn = gn*ones(1, params.maxit);

% INITIALIZATION
% primal variable ALREADY INITIALIZED
%w = params.w_init;
% dual variable
v1 = S_op(w_vec);  
v2 = D_op(w_vec); 

if nargout > 1 || params.verbosity > 1
    stat.f_eval = nan(params.maxit, 1);
    stat.g_eval1 = nan(params.maxit, 1);
    stat.g_eval2 = nan(params.maxit, 1);
    stat.g_eval = nan(params.maxit, 1);
    stat.h_eval = nan(params.maxit, 1);
    stat.fgh_eval = nan(params.maxit, 1);
    stat.pos_violation = nan(params.maxit, 1);
end
if params.verbosity > 1
    fprintf('Relative change of primal, dual variables, and objective fun\n');
end

tic
% gn = lin_map(params.step_size, [epsilon, (1-epsilon)/mu], [0,1]);              

for i = 1:params.maxit
    
    % added by wxl
    primal_gap_iter(i) = norm(w_vec-w_opt); % commented when comparing runtime
    
    Y_n = w_vec - gn(i) * (h.grad(w_vec) + St_op(v1) + Dt_op(v2));  % y^{(n)}
    y1 = v1 + gn(i)*(S_op(w_vec));  % \bar{y_1}
    y2 = v2 + gn(i)*(D_op(w_vec));  % \bar{y_2}
    P_n = f.prox(Y_n, gn(i)); % p^{(n)}
    p1 = g_star_prox1(y1, gn(i));  % \bar{p_1}
    p2 = g_star_prox2(y2, gn(i));  % \bar{p_2}
    Q_n = P_n - gn(i) * (h.grad(P_n) + St_op(p1) + Dt_op(p2));  % q_i
    q1 = p1 + gn(i) * (S_op(P_n));   % \bar{q_1}
    q2 = p2 + gn(i) * (D_op(P_n));   % \bar{q_1}
    
    if nargout > 1 || params.verbosity > 2
        stat.f_eval(i) = f.eval(w_vec);
        stat.g_eval1(i) = g.eval1(S_op(w_vec));
        stat.g_eval2(i) = g.eval2(D_op(w_vec));
        stat.g_eval(i) = stat.g_eval1(i) + stat.g_eval2(i);     
        stat.h_eval(i) = h.eval(w_vec);
        stat.fgh_eval(i) = stat.f_eval(i) + stat.g_eval(i) + stat.h_eval(i);
        stat.pos_violation(i) = -sum(min(0,w_vec));
    end  
    if params.verbosity > 3
        fprintf('iter %4d: %6.4e   %6.3e \n', i, rel_norm_primal, stat.fgh_eval(i));
    elseif params.verbosity > 2
        fprintf('iter %4d: %6.4e   %6.3e \n', i, rel_norm_primal, stat.fgh_eval(i));
    elseif params.verbosity > 1
        fprintf('iter %4d: %6.4e \n', i, rel_norm_primal);
    end
    
%     rel_norm_primal = norm(- Y_n + Q_n)/norm(w_vec);
    % modify
    vn = cat(1,v1,v2);
    yn = cat(1,y1,y2);
    qn = cat(1,q1,q2);
    rel_norm_primal = norm(- Y_n + Q_n, 'fro')/norm(w_vec, 'fro');
    rel_norm_dual = norm(- yn + qn)/norm(vn);
    
    w_vec = w_vec - Y_n + Q_n;
    v1 = v1 - y1 + q1;
    v2 = v2 - y2 + q2;
    
    % added by ycr
    if rel_norm_primal < params.tol && rel_norm_dual < params.tol
        iter = i;
        break
    end
end
stat.time = toc;
if params.verbosity > 0
    fprintf('# iters: %4d. Rel primal: %6.4e  OBJ %6.3e\n', i, rel_norm_primal, f.eval(w_vec) + g.eval1(S_op(w_vec)) + g.eval2(D_op(w_vec)) + h.eval(w_vec));
    fprintf('Time needed is %f seconds\n', stat.time);
end

%%
if params.verbosity > 3
    figure; plot(real([stat.f_eval, stat.g_eval, stat.h_eval])); hold all; plot(real(stat.fgh_eval), '.'); legend('f', 'g', 'h', 'f+g+h');
    figure; plot(stat.pos_violation); title('sum of negative (invalid) values per iteration')
    figure; semilogy(max(0,-diff(real(stat.fgh_eval'))),'b.-'); hold on; semilogy(max(0,diff(real(stat.fgh_eval'))),'ro-'); title('|f(i)-f(i-1)|'); legend('going down','going up');
end

w_all = reshape(w_vec, [num_edges, T]);

if params.fix_zeros
    w_full = zeros(num_edges, T);
    w_full(ind, 1:end) = w_all(:, 1:end);
    w_all = w_full;
end

if ndims(Z) == 3
    W_all = zeros(num_nodes, num_nodes, T);
    for ii=1:T
        W_all(:,:,ii) = squareform_sp(w_all(:,ii));
    end
else
    W_all = w_all;
end
fval_pds = stat.fgh_eval(min(iter,params.maxit));
end

function out = make_blkdiag_multi(S, T)
% Create sparse block daigonal matrix.
% S: The matrix you want to duplicate
% T; The number of matrix you want to duplicate.

[row, col, v] = find(S);
[N, E] = size(S);

row_idx = repmat(row, T, 1);
plus_ridx = repmat(0:N:N*(T-1), length(row), 1);
row_idx = row_idx + plus_ridx(:);

col_idx = repmat(col, T, 1);
plus_cidx = repmat(0:E:E*(T-1), length(col), 1);
col_idx = col_idx + plus_cidx(:);

value = repmat(v, T, 1);

out = sparse(row_idx, col_idx, value, N*T, E*T);

end

function [D, Dt] = make_diff_matrix(num_edge, T)
% Create sparse matrix D so that Dw=w-w_pre
% w = [w1' w2' ... wT']'  w_pre = [w1' w1' w2' ... w{T-1}']' 
%
%

sz_d = num_edge*T;

row_idx = num_edge+1 : sz_d;
col_idx_minus = 1 : sz_d-num_edge;
col_idx_plus = num_edge+1 : sz_d;

row_idx = [row_idx row_idx];
col_idx = [col_idx_minus col_idx_plus];
value = [-1*ones(1,length(col_idx_minus)) ones(1,length(col_idx_plus))];

D = sparse(row_idx, col_idx, value, sz_d, sz_d);
Dt = D.';
end