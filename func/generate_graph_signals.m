function X_noisy = generate_graph_signals(N,A,DIM)
% generate noisy signal
NUM = N;
D = diag(sum(full(A)));
L0 = D-full(A);
L0 = L0/trace(L0)*DIM;
[V,D] = eig(full(L0));
sigma = pinv(D);
mu = zeros(1,DIM);
gftcoeff = mvnrnd(mu,sigma,NUM);
X = V*gftcoeff';
X_noisy = X + 0.5*randn(size(X));