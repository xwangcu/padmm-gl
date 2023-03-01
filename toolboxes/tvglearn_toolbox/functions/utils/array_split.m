function [div_st, div_end] = array_split(T, nsections)
neach_section = floor(T/nsections);
extras = mod(T, nsections);
section_sizes = [0, repelem(neach_section+1, extras), repelem(neach_section, nsections-extras)];
div_point = cumsum(section_sizes);
div_st = zeros(nsections, 1);
div_end = zeros(nsections, 1);
for ii = 1:nsections
    div_st(ii) = 1 + div_point(ii);
    div_end(ii) = div_point(ii+1);
end
end