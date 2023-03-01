clear;
close all
rng(20);

% generate a graph
DIM = 20;
[A,XCoords, YCoords] = construct_graph(DIM,'gaussian', 0.75, 0.5);
% [A,XCoords, YCoords] = construct_graph(DIM,'er',0.2);
% [A,XCoords, YCoords] = construct_graph(DIM,'pa',1);

% generate graph signals
NUM = 100;
D = diag(sum(full(A)));
L0 = D-full(A);
L0 = L0/trace(L0)*DIM;
[V,D] = eig(full(L0));
sigma = pinv(D);
mu = zeros(1,DIM);
gftcoeff = mvnrnd(mu,sigma,NUM);
X = V*gftcoeff';
X_noisy = X + 0.5*randn(size(X));
Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       Z(i,j) = norm(X_noisy(i,:)-X_noisy(j,:),2)^2;
    end
end

%% common parameters
alpha = 200;
beta = 70;
max_iter = 1000;
epsilon = 1e-10;

%% obtain optimal solution via ADMM solver
fprintf('solving...\n');
t = 1;
tau1 = 1e-4;
tau2 = 1e-4;
tic
[W_opt] = gl_admm_solver(X_noisy, alpha, beta, t, tau1, tau2, 1e6, 1e-18);
fprintf('solved!\n');
toc
% load('W_opt.mat')

% %% CVX
% tic
% [W_cvx, ~] = gl_cvx(X_noisy, alpha, beta); % run algorithm
% cvx_time = toc;
% D = diag(sum(full(W_cvx)));
% L_cvx = D-full(W_cvx);
% L_cvx(abs(L_cvx)<10^(-4))=0;
% [precision_cvx, recall_cvx, Fmeasure_cvx, NMI_cvx, ~] = graph_learning_perf_eval(L0,L_cvx);
% fval_cvx = trace(W_cvx*Z) - alpha*sum(log(sum(W_cvx,2))) + 0.5*beta*(norm(W_cvx,'fro'))^2;

%% PDS
tau = 0.9;
Z0 = 1/sqrt(alpha*beta)*Z;
[W_pds, stat_pds, primal_gap_iter_pds] = gsp_learn_graph_log_degrees_(Z0, 1, 1, alpha, beta, tau, max_iter, epsilon, W_opt);
W_pds = sqrt(alpha/beta)*W_pds;

D = diag(sum(full(W_pds)));
L_pds = D-full(W_pds);
L_pds(abs(L_pds)<10^(-4))=0;
fval_pds = trace(W_pds*Z) - alpha*sum(log(sum(W_pds,2))) + 0.5*beta*(norm(W_pds,'fro'))^2;
[precision_pds, recall_pds, Fmeasure_pds, NMI_pds, num_of_edges_pds] = graph_learning_perf_eval(L0,L_pds);

%% ADMM
t = 190;
tau1 = 0.0001;
tau2 = 0.0005;
tic
[W_admm, fval_admm, primal_gap_iter_admm] = gl_admm(X_noisy, alpha, beta, t, tau1, tau2, max_iter, epsilon, W_opt);
admm_time = toc;
D = diag(sum(full(W_admm)));
L_admm = D-full(W_admm);
L_admm(abs(L_admm)<10^(-4))=0;
[precision_admm, recall_admm, Fmeasure_admm, NMI_admm, num_of_edges_admm] = graph_learning_perf_eval(L0,L_admm);

%% FDPG
reset = 50;
tic
[W_FDPG, fval_fdpg, primal_gap_iter_fdpg] = gl_fdpg(X_noisy, alpha, beta, max_iter, reset, W_opt);
fdpg_time = toc;
D = diag(sum(full(W_FDPG)));
L_fdpg = D-full(W_FDPG);
L_fdpg(abs(L_fdpg)<10^(-4))=0;
[precision_fdpg, recall_fdpg, Fmeasure_fdpg, NMI_fdpg, num_of_edges_fdpg] = graph_learning_perf_eval(L0,L_fdpg);

%% MM
tic
DIMw = DIM*(DIM-1)/2;
w_0 = ones(DIMw,1);
[w_mm, stat_mm, fval_mm, primal_gap_iter_mm] = MM_gl(X_noisy, alpha, beta, w_0, epsilon, NUM, max_iter, W_opt);
mm_time = toc;
W_mm  = linear_operator_vec2mat(w_mm, DIM);
D = diag(sum(full(W_mm)));
L_mm = D-full(W_mm);
L_mm(abs(L_mm)<10^(-4))=0;
[precision_mm, recall_mm, Fmeasure_mm, NMI_mm, num_of_edges_mm] = graph_learning_perf_eval(L0,L_mm);

%% outputs
fprintf('alpha=%.2f, beta=%.2f\n', alpha, beta);

% fprintf('----- CVX  Time needed is %f -----\n', cvx_time);
fprintf('----- PDS  Time needed is %f -----\n', stat_pds.time);
fprintf('----- ADMM Time needed is %f -----\n', admm_time);
fprintf('----- FDPG Time needed is %f -----\n', fdpg_time);
fprintf('----- MM Time needed is %f -----\n', mm_time);

% fprintf('CVX               | fval_cvx=%f\n', fval_cvx);
% fprintf('CVX measurements  | Fmeasure_cvx=%f, precision_cvx=%f, recall_cvx=%f, NMI_cvx=%.4f\n\n', Fmeasure_cvx, precision_cvx, recall_cvx, NMI_cvx);

fprintf('PDS               | fval_pds=%f \n', fval_pds);
fprintf('PDS measurements  | Fmeasure_pds=%f, precision_pds=%f, recall_pds=%f, NMI_pds=%.4f\n\n', Fmeasure_pds, precision_pds, recall_pds, NMI_pds);

fprintf('ADMM              | fval_admm=%f, t=%f, tau1=%f, tau2=%f, max_iter=%d\n', fval_admm(end), t, tau1, tau2, max_iter);
fprintf('ADMM measurements | Fmeasure_admm=%f, precision_admm=%f, recall_admm=%f, NMI_admm=%.4f\n\n', Fmeasure_admm, precision_admm, recall_admm, NMI_admm);
                         
fprintf('FDPG              | fval_fdpg=%f, max_iter=%d\n', fval_fdpg(end), max_iter);
fprintf('FDPG measurements | Fmeasure_fdpg=%f, precision_fdpg=%f, recall_fdpg=%f, NMI_fdpg=%.4f\n\n', Fmeasure_fdpg, precision_fdpg, recall_fdpg, NMI_fdpg);
                         
fprintf('MM              | fval_mm=%f, max_iter=%d\n', fval_mm(end), max_iter);
fprintf('MM measurements | Fmeasure_mm=%f, precision_mm=%f, recall_mm=%f, NMI_mm=%.4f\n\n', Fmeasure_mm, precision_mm, recall_mm, NMI_mm);
                         
%% figures
figure;
semilogy(primal_gap_iter_admm,'-r','LineWidth',1.5);
hold on;
semilogy(primal_gap_iter_pds,'-b','LineWidth',1.5,'LineStyle',"--");
hold on;
semilogy(primal_gap_iter_fdpg,'-g','LineWidth',1.5,'LineStyle',"-.");
hold on;
semilogy(primal_gap_iter_mm,'-k','LineWidth',1.5,'LineStyle',":");
hold on;
xlabel('iteration $k$','Interpreter','latex','FontSize',23);
ylabel('{$\|w-w^*\|_2$}','Interpreter','latex','FontSize',23);
lgd = legend('pADMM-GL','Primal-Dual','FDPG','MM','location','southwest');
lgd.FontSize = 15;
% title('Static Graph Learning','Interpreter','latex','FontSize',20);

% figure;
% plot(fval_admm,'-r','LineWidth',2);
% hold on;
% plot(fval_pds,'-b','LineWidth',2);
% hold on;
% plot(fval_fdpg,'-g','LineWidth',2);
% hold on;
% plot(fval_mm,'-k','LineWidth',2);
% hold on;
% xlabel('iteration','FontSize',30);
% ylabel('{$f(w)$}','Interpreter','latex','FontSize',30);
% lgd = legend('ADMM','Primal-Dual','FDPG','location','southwest');
% lgd.FontSize = 20;

% figure;
% subplot(2,2,1)
% imagesc(L0)
% colorbar
% title('Groundtruth')
% subplot(2,2,2)
% imagesc(L_cvx)
% colorbar
% title('CVX')
% subplot(2,2,3)
% imagesc(L_pds)
% colorbar
% title('PDS')
% subplot(2,2,4)
% imagesc(L_admm)
% colorbar
% title('ADMM')
