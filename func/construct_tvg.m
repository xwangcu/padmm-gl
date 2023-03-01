function [G, XCoords, YCoords] = construct_tvg(N,opt,varargin1,varargin2,varargin3)
% Graph construction
% N num of nodes
% varargin1 the probability of each edge p
% varargin2 num of time slots

%% check inputs
if nargin == 4
    if strcmp(opt,'tver') || strcmp(opt,'lfer')
        error('number of input variables not correct :(')
    end
end

%% generate coordinates of vertices
plane_dim = 1;
XCoords = plane_dim*rand(N,1);
YCoords = plane_dim*rand(N,1);

%% construct the graph
switch opt 
    case 'tver', % Erdos-Renyi random graph with temporal homogeneity
        p = varargin1;
        G = tvg_tver(N,varargin1,varargin2,varargin3);
       
    case 'lfer', % Erdos-Renyi random graph with switching behavior
        m = varargin1;
        G = preferential_attachment_graph(N,m);
end

%% plot the graph
% figure();wgPlot(G+diag(ones(N,1)),[XCoords YCoords],2,'vertexmetadata',ones(N,1));