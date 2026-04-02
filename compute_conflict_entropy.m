function [conflict, flags, threshold] = compute_conflict_entropy(BeliefCell, N_VIEWS)
%COMPUTE_CONFLICT_ENTROPY  Idea 2 — Pairwise Jensen-Shannon conflict entropy.
%
%  C(i) = sum_{a<b} JS(B_a(i,:), B_b(i,:))
%  Flag sample i as uncertain if C(i) > mean(C) + std(C).

T_test   = size(BeliefCell{1}, 1);
conflict = zeros(T_test,1);
pairs    = nchoosek(1:N_VIEWS, 2);

for i = 1:T_test
    ce = 0;
    for p = 1:size(pairs,1)
        P = BeliefCell{pairs(p,1)}(i,:);
        Q = BeliefCell{pairs(p,2)}(i,:);
        ce = ce + js_div(P, Q);
    end
    conflict(i) = ce;
end

threshold = mean(conflict) + std(conflict);
flags     = conflict > threshold;
n_unc     = sum(flags);

fprintf('\n========================================================\n');
fprintf('  Idea 2 - Cross-view Belief Conflict Entropy\n');
fprintf('========================================================\n');
fprintf('  Mean=%.4f  Std=%.4f  Threshold=%.4f\n',...
        mean(conflict), std(conflict), threshold);
fprintf('  Confident: %d/%d (%.1f%%)   Uncertain: %d/%d (%.1f%%)\n',...
        T_test-n_unc, T_test, (T_test-n_unc)/T_test*100,...
        n_unc, T_test, n_unc/T_test*100);
end

function d = js_div(P, Q)
eps_v = 1e-10;
P = real(P(:)') + eps_v;  P = P/sum(P);
Q = real(Q(:)') + eps_v;  Q = Q/sum(Q);
M = 0.5*(P+Q);
d = max(0, 0.5*sum(P.*log(P./M)) + 0.5*sum(Q.*log(Q./M)));
end
