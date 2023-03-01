function [W, fval_admm, primal_gap_iter] = gl_admm(X, alpha, beta, t, tau1, tau2, max_iter, epsilon, W_opt)

% min_{w,d} 2*z'*w - alpha*ones'*log(d) + beta*w'*w
% s.t.      Sw-d=0, w>=0

%% initialization
DIM = size(X,1);
DIMw = DIM*(DIM-1)/2;
Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       Z(i,j) = norm(X(i,:)-X(j,:),2)^2;
    end
end
w_opt =  squareform(W_opt)';
z = squareform(Z)'; % z = linear_operator_mat2vec(Z, DIM); 
% z = z/size(X,2); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[S, St] = sum_squareform(DIM);

%% iterations
% w = randn(DIMw,1);
w = ones(DIMw,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = randn(DIM,1);
y = randn(DIM,1);
fval_iter = zeros(max_iter,1);
primal_res_iter = zeros(max_iter,1);
dual_res_iter = zeros(max_iter,1);
primal_gap_iter = zeros(max_iter,1);

for k = 1 : max_iter
    fval_iter(k) = 2*(z')*w - alpha*sum(log(S*w)) + beta*(w')*w; % commented when comparing runtime
    
    % update w
    p = w - tau1*(t*St*(S*w-d) + 2*beta*w - St*y + 2*z);
    w = max(p, 0);
    
    % update d
    d_tmp = d;
    Sw = S*w;
    q = (1-tau2*t)*d + tau2*t*Sw - tau2*y;
    d = 0.5 * (q + sqrt(q.^2+4*alpha*tau2));
    
    % updata y
    y = y - t*(Sw - d);
    
    % suboptimality measurements
    primal_res_iter(k) = norm(Sw-d);
    dual_res_iter(k) = norm(t*St*(d-d_tmp));
    primal_gap_iter(k) = norm(w-w_opt); % commented when comparing runtime
    
    % stopping criterion
    if (primal_res_iter(k) < epsilon) && (dual_res_iter(k) < epsilon)
%         fprintf('primal_gap_iter(%d)=%f',k,primal_res_iter(k));
%         fprintf('dual_gap_iter(%d)=%f',k,dual_res_iter(k));
        break;
    end
end
fval_admm = fval_iter(1:k);

W  = linear_operator_vec2mat(w, DIM);
