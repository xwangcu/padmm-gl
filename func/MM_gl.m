%%%graph learning via MM

%%Inputs:
%X_noisy: noisy data matrix of size p*nSamples
%a and b: the hyperparameters alpha and beta
%w_0: initial value of weight vector w
%thresh: Convergence threshold to exit the loop
%nSamples: Number of realizations of graph signal

%Outputs:
%wk: the estimated weight vector
%stat: A structure array containing percentage of non-zero entities in %w,time, number of iterations and f(w) as fields.

function [wk, stat, fval_mm, primal_gap_iter] = MM_gl(X_noisy, a, b, w_0, thresh, nSamples, max_iter, W_opt)

w_opt =  squareform(W_opt)'; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
primal_gap_iter = zeros(max_iter,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = sparse(gsp_distanz(X_noisy').^2); %pairwise distance matrix
d = squareform_sp(D); %vectored Z
% D = (gsp_distanz(X_noisy').^2); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% d = squareform(D)'; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = length(d);  %m=(p*(p-1))/2;
p = round((1 + sqrt(1+8*m))/ 2); %number of nodes
[S, St] = sum_squareform(p); %S is a binary matrix such that Sw=W1, where W is the weight matrix; St is the transpose of S
wk=w_0;
wk(wk==0)=eps;
obj_val(1) = (2*wk'*d)+(b*(norm(wk))^2)-(a*sum(log(S*wk))); %objective function value
n_z(1)=length(find(wk~=0))*100/m; %Percentage of non-zero (active) elements in the weight vector w
tim(1)=0;
idx=2;
eta=1;
tic
d2=-2*d;
d4=4*(d.^2);
b8=8*b;
b4=4*b;
Swk=S*wk;
Swki=1./Swk;
Swkii=repmat(Swki,[1 m]);
SS=zeros(p,m); SS = sparse(SS);

% my code
% Sw = zeros(p,1);
% S_tmp = zeros(p,m);
fval_mm = zeros(max_iter,1);

for k = 1 : max_iter
    SS(S~=0)=Swkii(S~=0);
    cl=a*sum(SS)'.*wk;
    
    % my code
    fval_mm(k) = (2*wk'*d)+(b*(norm(wk))^2)-(a*sum(log(S*wk)));
    primal_gap_iter(k) = norm(wk-w_opt); % commented when comparing runtime
    fprintf('k =%d\n',k);
%     primal_gap_iter(k) = norm(wk-w_opt); % commented when comparing runtime
%     for i = 1 : p
%         Sw(i) = S(i,:) * wk;
%         S_tmp(i,:) = S(i,:)/Sw(i);
%     end
%     cl = a * sum(S_tmp,1)';
    
    u=((d2)+sqrt((d4)+(b8*cl)))/(b4);
    wk=u;
    wk(u<0.001)=0;
    n_z(idx)=length(find(wk~=0))*100/m;
    Swk=S*wk;
    Swki=1./Swk;
    Swkii=repmat(Swki,[1 m]);
    obj_val(idx) = (2*u'*d)+(b*(norm(u))^2)-(a*sum(log(S*u)));
    tim(idx)=toc;
    eta=abs(obj_val(idx-1)-obj_val(idx));
    idx=idx+1;
    
    % my code
    if primal_gap_iter(k) < thresh
        break;
    end
end

% my code
fval_mm = fval_mm(1:k);

% fprintf('\n epsilon = %f, iter = %g\n', eta, k);
stat.non_zero=n_z; %percentage of active elements in w
stat.time = tim(idx-1); %total time taken
stat.num_itr = length(obj_val); %total number of iterations
stat.obj_val = obj_val; %objective function value
end

