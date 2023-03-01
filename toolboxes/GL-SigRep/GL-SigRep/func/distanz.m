function d = distanz(X)
    n = size(X,2);
    d = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if i ~= j
                d(i,j) = norm(X(:,i)-X(:,j));
            end
        end
    end
end