function [W, obj_val_ours, primal_gap_iter] = gl_fdpg_runtime(X, a, b, itr, reset, W_opt, epsilon)

seed = 12345;
rng(seed);

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
[S, St] = sum_squareform(DIM);

primal_gap_iter = zeros(itr,1);
L = (DIM-1)/(b);
prox_g = @(x) (x+sqrt(x.^2+4*a*L))./2;

f.eval = @(w) b * norm(w)^2;
g.eval = @(w) 2 * w' * z - a * sum(log(S*w));

wk = rand(DIM,1);
yk = wk;
tk = 1;

tic
for k=1:itr
    u = max(eps,(St*wk-2*z)./(2*b));
%     v = prox_g((S*u-L*wk));
    v = (((S*u-L*wk))+sqrt(((S*u-L*wk)).^2+4*a*L))./2;
    yK = wk - (1/L).*(S*u-v);
    tK = (1+sqrt(1+4*(tk^2)))/2;
    
    wK = yK + ((tk-1)/tK)*(yK - yk);
    
    obj_val_ours(k) = 2 * u' * z - a * sum(log(S*u)) + b * norm(u)^2;
    
    tk = tK;
    yk = yK;
    wk = wK;
    
    w_hat = max(0, (St*yK - 2*z)./(2*b));
    if rem(k,reset) == 0
        tk = 1;
    end
    
    %%
    primal_gap_iter(k) = norm(w_hat - w_opt, 2);
    if primal_gap_iter(k) < epsilon
        break
    end
end
stat.time = toc;
stat.num_itr = length(obj_val_ours);
stat.obj_val = obj_val_ours;
W  = linear_operator_vec2mat(w_hat, DIM);
end