function L = optimize_laplacian_gaussian(N,Y,alpha,beta)

%% Laplacian constraints
[A1,b1,A2,b2,mat_obj] = laplacian_constraint_vech(N);
p = vec(Y*Y')';

%% optimization
cvx_begin

% cvx_solver mosek

variable L(N*(N+1)/2,1)

minimize alpha*p*mat_obj*L + beta*sum_square_abs(mat_obj*L)

subject to
    A1*L == b1
    A2*L <= b2

cvx_end

%% convert from vector form to matrix form
% L = convert_matrix(L,N);
L = reshape(mat_obj*L,N,N);

% %% difference vector
% diff = zeros(1,N*(N+1)/2);
% k = 1;
% for i = 1:N
%     for j = i:N
%         diff(k) = -norm(Y(i,:)-Y(j,:),2)^2;
%         k = k+1;
%     end
% end
% 
% %% matrix for constraint 1 (zero row-sum)
% for i = 1:N
%     tmp0{i} = zeros(1,N+1-i);
% end
% 
% mat_cons1 = zeros(N,N*(N+1)/2);
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
% %% matrix for constraint 2 (non-positive off-diagonal entries)
% for i = 1:N
%     tmp{i} = ones(1,N+1-i);
%     tmp{i}(1) = 0;
% end
% 
% mat_cons2 = diag(horzcat(tmp{:}));
% 
% %% vector for constraint 3 (trace constraint)
% vec_cons3 = ones(1,N*(N+1)/2)-horzcat(tmp{:});
% 
% %% matrix for objective
% mat_obj = zeros(N^2,N*(N+1)/2);
% 
% for i = 1:N
%     for j = 1:N
%         if j <= i-1
%             tmp = tmp0;
%             tmp{j}(i+1-j) = 1;
%             mat_obj((i-1)*N+j,:) = horzcat(tmp{:});
%         else
%             tmp = tmp0;
%             tmp{i}(j-i+1) = 1;
%             mat_obj((i-1)*N+j,:) = horzcat(tmp{:});
%         end
%     end
% end
% 
% %% optimization
% cvx_begin
% 
% % cvx_solver mosek
% 
% variable L(N*(N+1)/2,1)
% 
% minimize alpha*diff*L + beta*sum_square_abs(mat_obj*L)
% 
% subject to
%     mat_cons1*L == zeros(N,1)
%     mat_cons2*L <= 0
%     vec_cons3*L == N
% 
% cvx_end
% 
% %% convert from vector form to matrix form
% L = convert_matrix(L,N);
%