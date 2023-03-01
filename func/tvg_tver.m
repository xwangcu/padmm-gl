function [G, XCoords, YCoords] = tvg_tver(N,varargin1,varargin2,varargin3)
% Erdos-Renyi random graph with temporal homogeneity

% Time-Varying Graph Learning
% with Constraints on Graph Temporal Variation

% Graph construction
% N num of nodes
% varargin1 the probability of each edge p
% varargin2 num of time slots
% varargin3 portion of resampling

rng(30);
G = cell(varargin2,1);

%% generate coordinates of vertices
plane_dim = 1;
XCoords = plane_dim*rand(N,1);
YCoords = plane_dim*rand(N,1);

%% construct the graph

% base graph
p = varargin1;
G_1= double(triu(rand(N)<p,1));
index = G_1>0;
G_1 = sparse(G_1);
weights = rand(nnz(G_1),1);
G_1(index) = weights;
G{1}=G_1+G_1';

% num resample
nr = ceil(nnz(G_1)*varargin3);
for t=2:varargin2
    resample = randi([1,nnz(G_1)],nr,1);
    weights(resample) = rand(nr,1);
    G_1(index) = weights;
    G{t}=G_1+G_1';
end
%% plot the graph
% figure();wgPlot(G+diag(ones(N,1)),[XCoords YCoords],2,'vertexmetadata',ones(N,1));