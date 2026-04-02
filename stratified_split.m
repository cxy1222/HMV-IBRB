function [trainIdx, testIdx] = stratified_split(labels, train_ratio, seed)
%STRATIFIED_SPLIT  Stratified 80/20 split preserving class proportions.
rng(seed);
N_total  = length(labels);
trainIdx = false(N_total,1);
testIdx  = false(N_total,1);
classes  = unique(labels);
for c = 1:length(classes)
    idx  = find(abs(labels - classes(c)) < 1e-9);
    idx  = idx(randperm(length(idx)));
    n_tr = max(1, round(length(idx)*train_ratio));
    trainIdx(idx(1:n_tr))     = true;
    testIdx (idx(n_tr+1:end)) = true;
end
end
