function [G, XCoords, YCoords] = construct_graph(N,opt,varargin1,varargin2)
% Graph construction

%% check inputs
if nargin == 2
    if strcmp(opt,'chain') == 0
        error('number of input variables not correct :(')
    end
elseif nargin == 3
    if strcmp(opt,'gaussian') || strcmp(opt,'ff')
        error('number of input variables not correct :(')
    end
elseif nargin == 4
    if strcmp(opt,'er') || strcmp(opt,'pa')
        error('number of input variables not correct :(')
    end
end

%% generate coordinates of vertices
plane_dim = 1;
XCoords = plane_dim*rand(N,1);
YCoords = plane_dim*rand(N,1);

%% construct the graph
switch opt
    case 'gaussian', % random graph with Gaussian weights
        T = varargin1; 
        s = varargin2;
        d = distanz([XCoords,YCoords]'); 
        W = exp(-d.^2/(2*s^2)); 
        W(W<T) = 0; % Thresholding to have sparse matrix
        W = 0.5*(W+W');
        G = W-diag(diag(W));
        
    case 'er', % Erdos-Renyi random graph
        p = varargin1;
        G = erdos_reyni(N,p);
        
    case 'pa', % scale-free graph with preferential attachment
        m = varargin1;
        G = preferential_attachment_graph(N,m);
        
    case 'ff', % forest-fire model
        p = varargin1;
        r = varargin2;
        G = forest_fire_graph(N,p,r);
        
    case 'chain' % chain graph
        G = spdiags(ones(N-1,1),-1,N,N);
        G = G + G';
end

%% plot the graph
% figure();wgPlot(G+diag(ones(N,1)),[XCoords YCoords],2,'vertexmetadata',ones(N,1));