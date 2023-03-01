function [w] = linear_operator_L2w(L, m)

d = m*(m-1)/2;
w = zeros(d,1);

for i = 2 : m
    for j = 1 : i-1
        k = (j-1)*m - j*(j+1)/2 + i;
        fprintf('linear_operator_L2w: k=%d, i=%d, j=%d\n', k, i, j);
        w(k) = -L(i,j);
    end
end

end