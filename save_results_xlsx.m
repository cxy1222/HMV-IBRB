function save_results_xlsx(results, CQI, dim_scores, dim_names, ...
                           conflict, flags, threshold, best_view, ...
                           final_pred, true_labels, LABELS, results_dir, ...
                           testIdx, raw_data, dataset_name, good_views, fusion_min_acc)
%SAVE_RESULTS_XLSX  Export MetaBRB Gearbox results to Excel (9 sheets).
%
%  Gearbox fault label mapping:
%    0.00→Surface  0.25→Root  0.50→Miss  0.75→Health  1.00→Chipped
%
%  Sheet 1  Performance_Summary   : Acc/Prec/Rec/F1/AUC/MSE (all views + fused)
%  Sheet 2  PerClass_Metrics      : per-class Prec/Rec/F1/AUC per view
%  Sheet 3  Confusion_Matrices    : 5×5 confusion matrix for each view + fused
%  Sheet 4  ROC_AUC               : per-class AUC + Macro AUC per view
%  Sheet 5  View1_LDA_Pred        : sample predictions + belief columns
%  Sheet 6  View4_NCA_Pred
%  Sheet 7  View7_TSNE_Pred
%  Sheet 8  Conflict_Entropy      : conflict score per test sample
%  Sheet 9  CQI_Dimensions        : CQI five-dimension breakdown

if ~exist(results_dir,'dir'), mkdir(results_dir); end
fname   = fullfile(results_dir, 'MetaBRB_Gearbox_Results.xlsx');
N_VIEWS = numel(results);
T_test  = numel(true_labels(:));
N_OUT   = size(results(1).belief_out, 2);
N_CLS   = numel(LABELS);

true_labels = true_labels(:);
final_pred  = final_pred(:);
conflict    = conflict(:);
flags       = double(flags(:));

FAULT_NAMES = {'Surface','Root','Miss','Health','Chipped'};
lab_str     = arrayfun(@(v) sprintf('%.2f',v), LABELS, 'UniformOutput', false);

%% =====================================================================
%% Compute full classification metrics for each view + fused output
%% =====================================================================
view_metrics = cell(N_VIEWS+1, 1);    % last entry = fused
for v = 1:N_VIEWS
    view_metrics{v} = full_cls_metrics( ...
        results(v).test_pred(:), results(v).true_labels(:), ...
        results(v).belief_out, LABELS);
end
% Fused belief: use best-view belief for confident samples, weighted for uncertain
final_belief = results(best_view).belief_out;
mse_vals = arrayfun(@(v) results(v).test_mse, 1:N_VIEWS) + 1e-10;
mse_gated = mse_vals;  mse_gated(~good_views) = Inf;
rel_wts = (1./mse_gated) / sum(1./mse_gated(good_views));
if any(flags)
    unc_blf = zeros(sum(flags==1), N_CLS);
    for v = 1:N_VIEWS
        unc_blf = unc_blf + rel_wts(v) * results(v).belief_out(flags==1, :);
    end
    final_belief(flags==1, :) = unc_blf;
end
rs = sum(final_belief,2); rs(rs<1e-12)=1;
final_belief = bsxfun(@rdivide, final_belief, rs);
view_metrics{N_VIEWS+1} = full_cls_metrics(final_pred, true_labels, final_belief, LABELS);

all_view_names = [VIEW_NAMES_from_results(results), {'CQI+Conflict-Aware(Fused)'}];

%% =====================================================================
%% Sheet 1: Performance Summary (expanded with Prec/Rec/F1/AUC)
%% =====================================================================
hdr1 = {'View','Train_MSE','Test_MSE','Test_RMSE', ...
        'Train_Acc_pct','Test_Acc_pct', ...
        'MacroPrec','MacroRec','MacroF1','MacroAUC', ...
        'WtdPrec','WtdRec','WtdF1', ...
        'CQI','Is_Best_CQI','In_Fusion'};

n_rows = N_VIEWS + 1;   % views + fused
D1 = cell(n_rows, numel(hdr1));

for v = 1:N_VIEWS
    m = view_metrics{v};
    D1{v,1}  = results(v).view_name;
    D1{v,2}  = results(v).train_mse;
    D1{v,3}  = results(v).test_mse;
    D1{v,4}  = sqrt(results(v).test_mse);
    D1{v,5}  = results(v).train_acc;
    D1{v,6}  = results(v).test_acc;
    D1{v,7}  = m.macro_precision;
    D1{v,8}  = m.macro_recall;
    D1{v,9}  = m.macro_f1;
    D1{v,10} = m.macro_auc;
    D1{v,11} = m.weighted_precision;
    D1{v,12} = m.weighted_recall;
    D1{v,13} = m.weighted_f1;
    D1{v,14} = CQI(v);
    D1{v,15} = double(v==best_view);
    D1{v,16} = double(good_views(v));
end
% Fused row
mf = view_metrics{N_VIEWS+1};
fused_mse = mean((final_pred - true_labels).^2);
D1{n_rows,1}  = 'CQI+Conflict-Aware(Fused)';
D1{n_rows,3}  = fused_mse;
D1{n_rows,4}  = sqrt(fused_mse);
D1{n_rows,6}  = mean(classify_output(final_pred,LABELS)==true_labels)*100;
D1{n_rows,7}  = mf.macro_precision;
D1{n_rows,8}  = mf.macro_recall;
D1{n_rows,9}  = mf.macro_f1;
D1{n_rows,10} = mf.macro_auc;
D1{n_rows,11} = mf.weighted_precision;
D1{n_rows,12} = mf.weighted_recall;
D1{n_rows,13} = mf.weighted_f1;
D1{n_rows,14} = NaN; D1{n_rows,15} = NaN; D1{n_rows,16} = NaN;

writetable(cell2table(D1,'VariableNames',hdr1), fname, 'Sheet','Performance_Summary');
meta = {sprintf('Dataset: %s', dataset_name); ...
        sprintf('Fusion_Min_Acc: %.0f%%', fusion_min_acc); ...
        sprintf('Views_In_Fusion: %d/%d', sum(good_views), N_VIEWS); ...
        sprintf('Improvement: Class-sorted refs + Quality-gated fusion')};
writecell(meta, fname, 'Sheet','Performance_Summary','Range','R1');
fprintf('  [Sheet 1] Performance Summary  (Acc/Prec/Rec/F1/AUC)\n');

%% =====================================================================
%% Sheet 2: Per-Class Metrics
%% =====================================================================
hdr2 = {'View','Fault_Type','Class_Label','Support', ...
        'Precision','Recall','F1_Score','AUC'};
rows2 = {};
for v = 1:N_VIEWS+1
    m  = view_metrics{v};
    vn = all_view_names{v};
    for c = 1:N_CLS
        fn = '';
        if c <= numel(FAULT_NAMES), fn = FAULT_NAMES{c}; end
        rows2(end+1,:) = {vn, fn, lab_str{c}, m.support(c), ...
                          m.per_class_precision(c), m.per_class_recall(c), ...
                          m.per_class_f1(c), m.per_class_auc(c)}; %#ok
    end
    rows2(end+1,:) = {vn,'MACRO_AVG','',sum(m.support), ...
                      m.macro_precision, m.macro_recall, ...
                      m.macro_f1, m.macro_auc}; %#ok
    rows2(end+1,:) = {vn,'WEIGHTED_AVG','',sum(m.support), ...
                      m.weighted_precision, m.weighted_recall, ...
                      m.weighted_f1, ''}; %#ok
end
writetable(cell2table(rows2,'VariableNames',hdr2), fname, 'Sheet','PerClass_Metrics');
fprintf('  [Sheet 2] Per-Class Metrics  (Prec/Rec/F1/AUC per fault type)\n');

%% =====================================================================
%% Sheet 3: Confusion Matrices (raw counts + row-normalised %)
%% =====================================================================
cm_rows = {};
col_hdr = N_CLS + 1;   % True\Pred + N_CLS label cols
for v = 1:N_VIEWS+1
    m  = view_metrics{v};
    CM = m.confusion;
    vn = all_view_names{v};

    % View header
    cm_rows(end+1,:) = [{sprintf('=== %s ===', vn)}, repmat({''},1,N_CLS)]; %#ok

    % Fault-name header
    fault_hdr = [{'Fault'}, FAULT_NAMES(1:N_CLS)];
    cm_rows(end+1,:) = [{sprintf('True \\ Pred [%s]',vn)}, lab_str]; %#ok

    % Raw counts
    for ri = 1:N_CLS
        fn = ''; if ri<=numel(FAULT_NAMES), fn=FAULT_NAMES{ri}; end
        cm_rows(end+1,:) = [{sprintf('%s(%.2f)',fn,LABELS(ri))}, num2cell(CM(ri,:))]; %#ok
    end

    % Row-normalised %
    cm_rows(end+1,:) = [{'(% row-norm)'}, repmat({''},1,N_CLS)]; %#ok
    rs = sum(CM,2); rs(rs==0)=1;
    CM_pct = CM ./ rs * 100;
    for ri = 1:N_CLS
        fn = ''; if ri<=numel(FAULT_NAMES), fn=FAULT_NAMES{ri}; end
        cm_rows(end+1,:) = [{sprintf('%s(%.2f)',fn,LABELS(ri))}, ...
                             num2cell(round(CM_pct(ri,:),1))]; %#ok
    end
    cm_rows(end+1,:) = repmat({''},1,N_CLS+1); %#ok
end
writecell(cm_rows, fname, 'Sheet','Confusion_Matrices');
fprintf('  [Sheet 3] Confusion Matrices  (raw + row-normalised)\n');

%% =====================================================================
%% Sheet 4: ROC / AUC
%% =====================================================================
hdr4 = [{'View'}, ...
        arrayfun(@(c) sprintf('AUC_%s_%s', FAULT_NAMES{c}, lab_str{c}), ...
                 1:N_CLS, 'UniformOutput',false), ...
        {'Macro_AUC'}];
D4 = cell(N_VIEWS+1, numel(hdr4));
for v = 1:N_VIEWS+1
    m = view_metrics{v};
    D4{v,1} = all_view_names{v};
    for c = 1:N_CLS
        D4{v,c+1} = m.per_class_auc(c);
    end
    D4{v,N_CLS+2} = m.macro_auc;
end
writetable(cell2table(D4,'VariableNames',hdr4), fname, 'Sheet','ROC_AUC');
fprintf('  [Sheet 4] ROC / AUC  (per fault type + macro)\n');

%% =====================================================================
%% Sheets 5-7: Predictions per view (with belief columns)
%% =====================================================================
sheet_pred_names = {'View1_LDA_Pred','View4_NCA_Pred','View7_TSNE_Pred'};
for v = 1:N_VIEWS
    m  = results(v).metrics_ext;
    T  = length(m.y_true);
    B  = results(v).belief_out;

    fault_col = cell(T,1);
    for i=1:T
        idx = round(m.y_true(i)/0.25)+1;
        if idx>=1&&idx<=5, fault_col{i}=FAULT_NAMES{idx}; else, fault_col{i}='?'; end
    end

    base = array2table([(1:T)', m.y_true, m.y_class, ...
                         m.y_pred, abs(m.y_pred-m.y_true), ...
                         (m.y_pred-m.y_true).^2], ...
        'VariableNames',{'SampleID','True_Label','True_Class', ...
                         'Predicted','AbsError','SqError'});
    fault_tbl = cell2table(fault_col, 'VariableNames', {'Fault_Type'});
    belief_tbl = array2table(B, 'VariableNames', ...
        arrayfun(@(c) sprintf('Belief_%s',lab_str{c}), 1:N_CLS, 'UniformOutput',false));

    writetable([base(:,1), fault_tbl, base(:,2:end), belief_tbl], ...
               fname, 'Sheet', sheet_pred_names{v});
    fprintf('  [Sheet %d] %s predictions + beliefs\n', v+4, results(v).view_name);
end

%% =====================================================================
%% Sheet 8: Conflict Entropy
%% =====================================================================
fc = classify_output(final_pred, LABELS);
D8 = [(1:T_test)', true_labels, conflict, flags, final_pred, fc];
writetable(array2table(D8,'VariableNames', ...
    {'Sample_ID','True_Label','Conflict_Entropy','Uncertain_Flag','Final_Pred','Final_Class'}), ...
    fname,'Sheet','Conflict_Entropy');
writecell({'Threshold_mean+std', threshold; ...
           'N_Uncertain', sum(flags); ...
           'N_Confident', sum(~logical(flags)); ...
           'Dataset', dataset_name; ...
           'Fusion_Quality_Gate(%)', fusion_min_acc}, ...
    fname,'Sheet','Conflict_Entropy','Range','H1');
fprintf('  [Sheet 8] Conflict Entropy\n');

%% =====================================================================
%% Sheet 9: CQI Dimensions
%% =====================================================================
hdr9 = [{'View'}, dim_names(:)', {'CQI','Is_Best','In_Fusion'}];
rows9 = cell(N_VIEWS, numel(hdr9));
for v = 1:N_VIEWS
    rows9{v,1} = results(v).view_name;
    for d = 1:5, rows9{v,d+1} = dim_scores(v,d); end
    rows9{v,7} = CQI(v);
    rows9{v,8} = double(v==best_view);
    rows9{v,9} = double(good_views(v));
end
writetable(cell2table(rows9,'VariableNames',hdr9), fname,'Sheet','CQI_Dimensions');
fprintf('  [Sheet 9] CQI Dimensions\n');

fprintf('  [DONE] %s  (9 sheets)\n', fname);
end

%% ======================================================================
%%  HELPER: extract view names from results struct array
%% ======================================================================
function names = VIEW_NAMES_from_results(results)
n = numel(results);
names = cell(1, n);
for v = 1:n
    names{v} = results(v).view_name;
end
end

%% ======================================================================
%%  HELPER: compute full classification metrics
%%    Acc / per-class Prec,Rec,F1 / Macro+Weighted avg / Confusion / ROC-AUC
%% ======================================================================
function metrics = full_cls_metrics(y_pred, y_true, belief, LABELS)
N_CLS = numel(LABELS);
T     = numel(y_true);
key   = round(LABELS * 1e6);

y_cls   = classify_output(y_pred, LABELS);
y_cls   = y_cls(:);
y_true  = y_true(:);
belief  = max(belief, 0);
rs      = sum(belief,2); rs(rs<1e-12)=1;
belief  = bsxfun(@rdivide, belief, rs);

% Map to 1-based class indices
true_idx = zeros(T,1);
pred_idx = zeros(T,1);
for i = 1:T
    [~,ti] = min(abs(round(y_true(i)*1e6) - key));
    [~,pi] = min(abs(round(y_cls(i) *1e6) - key));
    true_idx(i) = ti;
    pred_idx(i) = pi;
end

% Confusion matrix (row=true, col=predicted)
CM = zeros(N_CLS, N_CLS);
for i = 1:T
    CM(true_idx(i), pred_idx(i)) = CM(true_idx(i), pred_idx(i)) + 1;
end

% Per-class Precision / Recall / F1
precision = zeros(N_CLS,1);
recall    = zeros(N_CLS,1);
f1        = zeros(N_CLS,1);
support   = sum(CM,2);
for c = 1:N_CLS
    TP = CM(c,c);
    FP = sum(CM(:,c)) - TP;
    FN = sum(CM(c,:)) - TP;
    p  = TP / (TP + FP + eps);
    r  = TP / (TP + FN + eps);
    precision(c) = p;
    recall(c)    = r;
    d = p + r;
    if d >= eps, f1(c) = 2*p*r/d; end
end

total = max(sum(support), 1);
metrics.accuracy           = mean(y_cls == y_true) * 100;
metrics.macro_precision    = mean(precision);
metrics.macro_recall       = mean(recall);
metrics.macro_f1           = mean(f1);
metrics.weighted_precision = sum(precision .* support) / total;
metrics.weighted_recall    = sum(recall    .* support) / total;
metrics.weighted_f1        = sum(f1        .* support) / total;
metrics.per_class_precision = precision;
metrics.per_class_recall    = recall;
metrics.per_class_f1        = f1;
metrics.support             = support;
metrics.confusion           = CM;

% ROC / AUC (one-vs-rest, belief as score)
roc_fpr = cell(N_CLS,1);
roc_tpr = cell(N_CLS,1);
roc_auc = zeros(N_CLS,1);
for c = 1:N_CLS
    scores = belief(:,c);
    y_bin  = double(true_idx == c);
    if sum(y_bin)==0 || sum(y_bin)==T
        roc_fpr{c}=[0;1]; roc_tpr{c}=[0;1]; roc_auc(c)=0.5; continue;
    end
    [fv,tv,av] = local_roc(y_bin, scores);
    roc_fpr{c}=fv; roc_tpr{c}=tv; roc_auc(c)=av;
end
metrics.roc_fpr       = roc_fpr;
metrics.roc_tpr       = roc_tpr;
metrics.per_class_auc = roc_auc;
metrics.macro_auc     = mean(roc_auc);
end

function [fpr,tpr,auc_val] = local_roc(y_bin, scores)
[~,sidx] = sort(scores,'descend');
ys = y_bin(sidx);
Np = sum(y_bin); Nn = numel(y_bin)-Np;
tpts = [0; cumsum( ys)/(Np+eps)];
fpts = [0; cumsum(1-ys)/(Nn+eps)];
[fu,ui] = unique(fpts,'last'); tu = tpts(ui);
if fu(1)>0,   fu=[0;fu];   tu=[0;tu];   end
if fu(end)<1, fu=[fu;1];   tu=[tu;1];   end
fpr=fu(:); tpr=tu(:);
auc_val = max(0,min(1,trapz(fpr,tpr)));
end

function y_class = classify_output(y_cont, label_set)
y_cont    = y_cont(:);
label_set = label_set(:)';
T         = numel(y_cont);
y_class   = zeros(T,1);
for i = 1:T
    [~,k]      = min(abs(y_cont(i) - label_set));
    y_class(i) = label_set(k);
end
end
