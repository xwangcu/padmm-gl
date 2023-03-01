function [L,Y,L_harvard] = graph_learning_gaussian(X_noisy,param)
% Learning graphs (Laplacian) from structured signals
% Signals X follow Gaussian assumption

N = param.N;
max_iter = param.max_iter;
alpha = param.alpha;
beta = param.beta;

objective = zeros(max_iter,1);
Y_0 = X_noisy;
Y = Y_0;
for i = 1:max_iter
    
    % Step 1: given Y, update L
    L = optimize_laplacian_gaussian(N,Y,alpha,beta);
%     L = optimize_laplacian_gaussian_admm(N,Y,alpha,beta,0.1,1.5);

    % solution in the Harvard paper
    if i == 1
        L_harvard = L;
    end
        
    % Step 2: Given L, update Y
    % Y = (eye(N)+alpha*L)^(-1)*Y_0;
    R = chol(eye(N) + alpha*L);
    Y = R \ (R' \ (Y_0));
    
    % plot the objective
    % objective(i) = norm(Y-Y_0,'fro')^2 + alpha*trace(Y'*L*Y) + beta*(norm(L,'fro')^2);
    objective(i) = norm(Y-Y_0,'fro')^2 + alpha*vec(Y*Y')'*vec(L) + beta*(norm(L,'fro')^2);
    figure(3)
    plot(i,objective(i), '.r');
    hold on, drawnow
    
    % stopping criteria
    if i>=2 && abs(objective(i)-objective(i-1))<10^(-4)
        break
    end
    
end