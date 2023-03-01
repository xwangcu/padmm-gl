function [A1,b1,A2,b2,mat_obj] = laplacian_constraint_vech(N)
% constraints        
% mat_cons1*L == zeros(N,1)
% mat_cons2*L <= 0
% vec_cons3*L == N

% %% matrix for constraint 1 (zero row-sum)
% for i = 1:N
%     tmp0{i} = sparse(1,N+1-i);
% end
% 
% mat_cons1 = sparse(N,N*(N+1)/2);
% 
% for i = 1:N
%     
%     tmp = tmp0;
%     tmp{i} = tmp{i}+1;
%     for j = 1:i-1
%         tmp{j}(i+1-j) = 1;
%     end
%     
%     mat_cons1(i,:) = horzcat(tmp{:});
%     
% end
% 
% % for i = 1:N
% %     mat_cons1(i,N*i-N+i-(i*(i-1)/2):N*i-(i*(i-1)/2)) = ones(1,N-i+1);
% % end
% % 
% % for i = 1:N-1
% %     xidx = i+1:N;
% %     yidx = i*(N+N-(i-1))/2-(N-i-1):i*(N+N-(i-1))/2-(N-i-1)+N-i-1;
% %     mat_cons1(sub2ind(size(mat_cons1),xidx,yidx)) = 1;
% % end
% 
% %% matrix for constraint 2 (non-positive off-diagonal entries)
% for i = 1:N
%     tmp{i} = ones(1,N+1-i);
%     tmp{i}(1) = 0;
% end
% 
% mat_cons2 = spdiags(horzcat(tmp{:})',0,N*(N+1)/2,N*(N+1)/2);
% 
% %% vector for constraint 3 (trace constraint)
% vec_cons3 = sparse(ones(1,N*(N+1)/2)-horzcat(tmp{:}));
% 
% %% matrix for objective
% % mat_obj = sparse(N^2,N*(N+1)/2);
% % 
% % for i = 1:N
% %     for j = 1:N
% %         if j <= i-1
% %             tmp = tmp0;
% %             tmp{j}(i+1-j) = 1;
% %             mat_obj((i-1)*N+j,:) = horzcat(tmp{:});
% %         else
% %             tmp = tmp0;
% %             tmp{i}(j-i+1) = 1;
% %             mat_obj((i-1)*N+j,:) = horzcat(tmp{:});
% %         end
% %     end
% % end
% 
% mat_obj = vech2vec(N);
% 
% %% create constraint matrices
% % equality constraint A2*vech(L)==b2
% A1 = [mat_cons1;vec_cons3];
% b1 = [sparse(N,1);N];
% 
% % inequality constraint A1*vech(L)<=b1
% A2 = mat_cons2;
% b2 = sparse(N*(N+1)/2,1);



%% matrix for objective (vech -> vec)
mat_obj = DuplicationM(N);

%% matrix for constraint 1 (zero row-sum)
% B = sparse(N,N^2);
% for i = 1:N
%     B(i,(i-1)*N+1:i*N) = ones(1,N);
% end
X = ones(N);
[r,c] = size(X);
i     = 1:numel(X);
j     = repmat(1:c,r,1);
B     = sparse(i',j(:),X(:))';
mat_cons1 = B*mat_obj;

%% matrix for constraint 2 (non-positive off-diagonal entries)
for i = 1:N
    tmp{i} = ones(1,N+1-i);
    tmp{i}(1) = 0;
end
mat_cons2 = spdiags(horzcat(tmp{:})',0,N*(N+1)/2,N*(N+1)/2);

%% vector for constraint 3 (trace constraint)
vec_cons3 = sparse(ones(1,N*(N+1)/2)-horzcat(tmp{:}));

%% create constraint matrices
% equality constraint A2*vech(L)==b2
A1 = [mat_cons1;vec_cons3];
b1 = [sparse(N,1);N];

% inequality constraint A1*vech(L)<=b1
A2 = mat_cons2;
b2 = sparse(N*(N+1)/2,1);