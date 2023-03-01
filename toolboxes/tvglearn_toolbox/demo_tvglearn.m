clearvars; clc; close all;
addpath(genpath('./functions'));
addpath(genpath('./data'));

%------user setting-----------
% Dataset type can be selected from select 'TVRW', 'TVER', and 'LFER'
dataset_type = 'TVER'; 
% Method type can be selected from 'normal', 'tikhonov', 'fusedlasso', and 'grouplasso'
method_type = 'fusedlasso'; 
% hyperparameters
alpha = 2;
beta = 1;
eta = 10;
%-----------------------------

%% create distance matrix
dataset_name = strcat('dataset_', dataset_type);
load(dataset_name, 'signal', 'graph');

% create distance matrix
calc_Z = @(X) 1/(size(X, 2)) * squareform(pdist(X).^2);
[num_nodes, T, num_sample] = size(signal);
Z = zeros(num_nodes, num_nodes, T);
num_edges = num_nodes*(num_nodes-1)/2;
w_gtruth = zeros(num_edges, T);
z_all = zeros(num_edges, T);
for ii = 1:T
    X = squeeze(signal(:,ii,:));
    Z = calc_Z(X);
    w_gtruth(:, ii) = squareform_sp(graph(ii).W);
    z_all(:, ii) = squareform_sp(Z);
end

%%
switch method_type
    case 'normal'
        [w_all, ~] = tvglearn_normal(z_all, alpha, beta);
    case 'tikhonov'
        [w_all, ~] = tvglearn_tikhonov(z_all, alpha, beta, eta);
    case 'fusedlasso'
        [w_all, ~] = tvglearn_fusedlasso_real(z_all, alpha, beta, eta);
    case 'grouplasso'
        [w_all, ~] = tvglearn_grouplasso(z_all, alpha, beta, eta);
end
w_all(w_all<1e-3) = 0;
%%

idx_range = 1:50;
figure, imagesc(w_gtruth(idx_range,:));
figure, imagesc(w_all(idx_range,:)); colorbar;
