clear;close all
%% Generate a graph
nreplicate = 50; % repeat the same experiment (based on different graphs)
param.N = 20;
[A,XCoords, YCoords] = construct_graph(param.N,'gaussian',0.75,0.5);
% [A,XCoords, YCoords] = construct_graph(param.N,'er',0.2);
% [A,XCoords, YCoords] = construct_graph(param.N,'pa',1);

%% Generate the graph Laplacian 
L_0 = full(sgwt_laplacian(A,'opt','raw'));
L_0 = L_0/trace(L_0)*param.N;

%% generate training signals
[V,D] = eig(full(L_0));
sigma = pinv(D);
mu = zeros(1,param.N);
num_of_signal = 80;
avg = rand(param.N,1) * 1;
noise_level = 1;

%% set parameters
param.max_iter = 50;
% alpha = 10.^[-1:-0.1:-3];
% beta = 10.^[0:-0.1:-2];
% lambda = 10.^[3:-0.05:0];
alpha = 10.^(-2);
beta = 10.^(-0.2);
lambda = 10.^1;

precision = zeros(length(alpha),length(beta));
recall = zeros(length(alpha),length(beta));
Fmeasure = zeros(length(alpha),length(beta));
NMI = zeros(length(alpha),length(beta));
num_of_edges = zeros(length(alpha),length(beta));

%% main loop
for i = 1:length(alpha)
    for j = 1:length(beta)
        param.alpha = alpha(i);
        param.beta = beta(j);
        % GL-SigRep
        for ii = 1 : nreplicate
            gftcoeff = mvnrnd(mu,sigma,num_of_signal);
            X = V*gftcoeff'+avg;
            X_noisy = X + noise_level*randn(size(X));
            [L,Y,~] = graph_learning_gaussian(X_noisy,param);
            Lcell{i,j} = L;
            L(abs(L)<10^(-4))=0;
            [precision(i,j),recall(i,j),Fmeasure(i,j),NMI(i,j),num_of_edges(i,j)] = graph_learning_perf_eval(L_0,L);
            
            %% performance
            result1(:,:,ii) = precision;
            result2(:,:,ii) = recall;
            result3(:,:,ii) = Fmeasure;
            result4(:,:,ii) = NMI;
            result5(:,:,ii) = num_of_edges;
            
            data{ii,1} = A;
            data{ii,2} = L_0;
            data{ii,3} = X;
            data{ii,4} = X_noisy;
            
            graph_original{ii} = L_0;
            graph{ii} = Lcell;
        end
    end
end