function [M] = make_mlop(level)
T = 2^(level);
M = zeros(T, 2^(level+1)-1);
M(:, 1) = 1;
for l = 1 : level
    num_partitions = 2^(l);
    [div_st, div_end] = array_split(T, num_partitions);
    for m = 0:2^(l)-1
        index = 2^(l)+m;
        M(div_st(m+1):div_end(m+1), index) = 1;
    end
end
M = sparse(M);
end