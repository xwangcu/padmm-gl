function [M] = linear_operator_vec2mat(v, m)

d = size(v,1);
% fprintf('linear_operator_w2L: m = %d, d=%d\n', m, d);
if d ~= m*(m-1)/2
    if mod(d,m*(m-1)/2) == 0
        T = d/(m*(m-1)/2);
        M = cell(T,1);
        for l = 1:T
            M_tmp = zeros(m,m);
            v_tmp = v((l-1)*(m*(m-1)/2)+1:l*(m*(m-1)/2));
            for i = 1 : m
                for j = 1 : m
                    if i > j
                        k = i - j + 0.5*(j-1)*(2*m-j);
                        M_tmp(i,j) = v_tmp(k);
                    elseif i < j
                        k = j - i + 0.5*(i-1)*(2*m-i);
                        M_tmp(i,j) = v_tmp(k);
                    else
                        M_tmp(i,i) = 0;
                    end
                end
            end
            M{l} = M_tmp;
        end
    else
        fprintf('linear_operator_w2L: linear_operator size mismatch\n')
    end
else
    M = zeros(m,m);

    for i = 1 : m
        for j = 1 : m
            if i > j
                k = i - j + 0.5*(j-1)*(2*m-j);
                M(i,j) = v(k);
            elseif i < j
                k = j - i + 0.5*(i-1)*(2*m-i);
                M(i,j) = v(k);
            else
                M(i,i) = 0;
            end
        end
    end
end

end