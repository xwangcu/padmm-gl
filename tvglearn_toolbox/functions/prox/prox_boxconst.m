function[out] = prox_boxconst(x, x_lower, x_upper)
% This function computes the prox of box constraint
x(x < x_lower) = x_lower;
x(x > x_upper) = x_upper;
out = x;
end