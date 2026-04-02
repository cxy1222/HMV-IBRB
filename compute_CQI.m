function [CQI, dim_scores, best_view, dim_names] = compute_CQI(results, N_VIEWS)
%COMPUTE_CQI  Idea 1 — Five-dimensional Composite Quality Index.
%
%  Dimensions (all in [0,1], higher = better):
%    D1 Accuracy      (w=0.30)  Test classification accuracy
%    D2 Determinacy   (w=0.25)  Mean max-belief per test sample
%    D3 Generalisation(w=0.20)  1 - relative train-test MSE gap
%    D4 Rule Coverage (w=0.10)  Uniformity of rule activation counts
%    D5 Pred Quality  (w=0.15)  Inverse-normalised test MSE

dim_names = {'Accuracy','Determinacy','Generalisation','Rule Coverage','Pred. Quality'};
W         = [0.30, 0.25, 0.20, 0.10, 0.15];

dim_scores = zeros(N_VIEWS, 5);
all_mse    = arrayfun(@(v) results(v).test_mse, 1:N_VIEWS);
max_mse    = max(all_mse) + 1e-12;

for v = 1:N_VIEWS
    % D1
    dim_scores(v,1) = results(v).test_acc / 100;
    % D2
    B = results(v).belief_out;
    dim_scores(v,2) = mean(max(B,[],2));
    % D3
    tr = results(v).train_mse; te = results(v).test_mse;
    dim_scores(v,3) = max(0, 1 - abs(tr-te)/(tr+te+1e-12));
    % D4
    rc = results(v).rule_counts(:);
    if mean(rc) > 0
        cv = std(rc)/(mean(rc)+eps);
        dim_scores(v,4) = max(0, 1-min(cv,1));
    end
    % D5
    dim_scores(v,5) = 1 - results(v).test_mse/max_mse;
end

CQI = (dim_scores * W(:));   % N_VIEWS x 1
[~, best_view] = max(CQI);

fprintf('\n========================================================\n');
fprintf('  Idea 1 - Composite Quality Index (CQI)\n');
fprintf('========================================================\n');
fprintf('  Weights: ACC=%.2f DET=%.2f GEN=%.2f COV=%.2f PQ=%.2f\n', W);
fprintf('  %-14s %7s %7s %7s %7s %7s %7s\n',...
        'View','ACC','DET','GEN','COV','PQ','CQI');
fprintf('  %s\n', repmat('-',1,60));
for v = 1:N_VIEWS
    mk=''; if v==best_view, mk=' <<< Best'; end
    fprintf('  %-14s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f%s\n',...
            results(v).view_name, dim_scores(v,:), CQI(v), mk);
end
fprintf('\n  Best view: %s  (CQI=%.4f)\n',...
        results(best_view).view_name, CQI(best_view));
end
