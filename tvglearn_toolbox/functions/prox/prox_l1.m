function[out] = prox_l1(x, gamma)

x(abs(x) <= gamma) = 0;
x(abs(x) > gamma) = x(abs(x) > gamma) - sign(x(abs(x) > gamma))*gamma;
out = x;

end