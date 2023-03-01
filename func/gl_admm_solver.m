function [W] = gl_admm_solver(X, alpha, beta, t, tau1, tau2, max_iter, epsilon)

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
z = squareform(Z)'; % z = linear_operator_mat2vec(Z, DIM);
[S, St] = sum_squareform(DIM);

%% iterations
w = randn(DIMw,1);
d = randn(DIM,1);
y = randn(DIM,1);
primal_gap_iter = zeros(max_iter,1); 
dual_gap_iter = zeros(max_iter,1);

for k = 1 : max_iter
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
    primal_gap_iter(k) = norm(t*St*(d-d_tmp));
    dual_gap_iter(k) = norm(Sw-d);
    
    % stopping criterion
    if (primal_gap_iter(k) < epsilon) && (dual_gap_iter(k) < epsilon)
        fprintf('primal_gap_iter(%d)=%f',k,primal_gap_iter(k));
        fprintf('dual_gap_iter(%d)=%f',k,dual_gap_iter(k));
        break;
    end
end

W  = linear_operator_vec2mat(w, DIM);
