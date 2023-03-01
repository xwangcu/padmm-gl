function adj = preferential_attachment_graph(n,m)
%% create preferential attachment graph
edges = preferential_attachment(n,m);
adj = zeros(n);
for i = 1:size(edges,1)
    adj(edges(i,1),edges(i,2)) = 1;
end