function [W, fval_admm, fval_iter, primal_gap_iter] = dgl_admm(X, alpha, beta, gamma, t, tau1, tau2, max_iter, epsilon, w_opt,T)

% min_{w,v} 2*v'*w + beta*w'*w - alpha*ones'*log(v_1)+ gamma*|v_2|_{l1}
% s.t.      Q_dw-v=0, w>=0

%% initialization
[DIM,NUM] = size(X); % m in paper
DIMw = DIM*(DIM-1)/2; % p in paper
NUM = NUM/T;
% Q_1
[Qs, Qst] = sum_squareform(DIM);
QCell = repmat({Qs}, 1, T);
Q_1 = blkdiag(QCell{:});
Q_1t = Q_1';
% Q_2
[tm,tp] = size(Q_1);
% construct step by step
Q_211 = zeros(DIMw,tp);
Q_212 = -speye(tp-DIMw,tp);
Q_21 = cat(1,Q_211,Q_212);
Q_22 = speye(tp);
Q_22(1:DIMw,:) = 0;
Q_2 = Q_21 + Q_22;
Q_2t = Q_2';
% Q_d
Q_d = cat(1,Q_1,Q_2);
Q_dt = Q_d';

Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
        Z(i,j) = norm(X(i,1:NUM)-X(j,1:NUM),2)^2;
    end
end
z = squareform(Z)';

for k = 1:T-1
    Z = zeros(DIM,DIM);
    for i = 1 : DIM
        for j = 1 : DIM
            Z(i,j) = norm(X(i,1+k*NUM:(k+1)*NUM)-X(j,1+k*NUM:(k+1)*NUM),2)^2;
        end
    end
    z_new = squareform(Z)';
    z = cat(1,z,z_new);
end


%% iterations
w = zeros(tp,1);
v = randn(tm+tp,1);
y = randn(tm+tp,1);
fval_iter = zeros(max_iter,1);
primal_res_iter = zeros(max_iter,1); 
dual_res_iter = zeros(max_iter,1);
primal_gap_iter = zeros(max_iter,1);

% for varying t
% uncomment the below for varying penaly param
% miu = 10;
% tau_incr = 2;
% tau_decr = 2;
for k = 1 : max_iter
    
    % added by wxl // commented when comparing runtime
    primal_gap_iter(k) = norm(w-w_opt);
    
    % update w
    p = w - tau1*(t*Q_dt*(Q_d*w-v) + 2*beta*w - Q_dt*y + 2*z);
%     p = w - tau1*Q_dt*(Q_d*w-v-y/t) - 2*tau1*z;
    w = max(p, 0);
    
    % update v
    v_tmp = v;
    Qw = Q_d*w;
    q = (1-tau2*t)*v + tau2*t*Qw - tau2*y;
    v_1_tmp = q(1:tm);
    v_2_tmp = q(tm+1:tm+tp);
    v_1 = 0.5 * (v_1_tmp + sqrt(v_1_tmp.^2+4*alpha*tau2));
    v_2 = sign(v_2_tmp).*(max(abs(v_2_tmp)-tau2*gamma,0)); % 'gamma' added by wxl
    v = cat(1,v_1,v_2);

    % updata y
    y = y - t*(Qw - v);
    
    % suboptimality measurements
    primal_res_iter(k) = norm(Qw-v);
    dual_res_iter(k) = norm(t*Q_dt*(v-v_tmp));
    
    % uncomment the below for varying penalty param
%     if primal_res_iter(k)>miu*dual_res_iter(k)
%         t = tau_incr*t;
%     else
%         if dual_res_iter(k) >miu*primal_res_iter(k)
%             t = t/tau_decr;
%         end
%     end
    
    fval_iter(k) = 2*(z')*w - alpha*sum(log(v_1)) + gamma*norm(v_2,1) + beta*(w')*w; % commented when comparing runtime
    
    % stopping criterion
    if (primal_res_iter(k) < epsilon) && (dual_res_iter(k) < epsilon)
        fprintf('primal_gap_iter(%d)=%f',k,primal_res_iter(k));
        fprintf('dual_gap_iter(%d)=%f\n',k,dual_res_iter(k));
        fval_admm = fval_iter(k);
        break;
    end
end
if k == max_iter
    fval_admm = fval_iter(max_iter);
end
W  = linear_operator_vec2mat(w, DIM);
