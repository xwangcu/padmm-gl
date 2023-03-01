function adj = forest_fire_graph(n,p,r)
%% create forest fire graph
edges = forestFireModel(n,p,r);
adj = zeros(n);
for i = 1:n
    adj(i,edges{i}) = 1;
end
adj = makedouble(adj,'2');