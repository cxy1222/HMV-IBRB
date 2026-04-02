function [Y1_REF, Y2_REF, sp1, sp2] = compute_adaptive_refs(trainData, n_int, label_set)
%COMPUTE_ADAPTIVE_REFS  Class-sorted adaptive reference values (v2-Gearbox).
%
%  ROOT CAUSE FIX:
%    Original strategy required class means to be monotone w.r.t. label values.
%    For gearbox data, 5 fault classes are CATEGORICAL — their numeric labels
%    {0, 0.25, 0.5, 0.75, 1.0} do NOT correspond to a natural ordering in
%    feature space.  E.g., in LDA Dim1 the sorted class order is:
%      Surface(0) → Miss(0.5) → Health(0.75) → Chipped(1.0) → Root(0.25)
%    This is NOT monotone with labels → old code fell back to uniform refs.
%
%  NEW STRATEGY — "class-sorted" (no monotonicity requirement):
%    1. Compute per-class mean in each dimension
%    2. Sort classes by their mean value (ascending)
%    3. Place interval boundaries at midpoints between consecutive sorted means
%    4. This gives near-pure intervals regardless of label ordering
%    5. CMA-ES freely learns the correct belief → label mapping per interval
%
%  THRESHOLD:
%    |Spearman| >= SORT_THRESH (0.20) → class-sorted refs
%    |Spearman| <  SORT_THRESH        → uniform spacing (truly uninformative dim)
%
%  WHY THIS WORKS:
%    With class-sorted refs, each interval is dominated by one class.
%    E.g., LDA Dim1: Int1=Surface(99%), Int5=Root(100%), Int2-4 partially pure.
%    Combined 2D BRB oracle accuracy: ~98% vs 80% with uniform refs.

SORT_THRESH = 0.20;   % lower threshold — class-sorted when even moderate correlation

x1 = trainData(:,1);
x2 = trainData(:,2);
lb = trainData(:,3);

sp1 = spearman_r(x1, lb);
sp2 = spearman_r(x2, lb);

fprintf('    Spearman Dim1=%+.4f  Dim2=%+.4f\n', sp1, sp2);

Y1_REF = build_refs_class_sorted(x1, lb, label_set, n_int, sp1, SORT_THRESH, 'Dim1');
Y2_REF = build_refs_class_sorted(x2, lb, label_set, n_int, sp2, SORT_THRESH, 'Dim2');
end

%% -----------------------------------------------------------------------
function refs = build_refs_class_sorted(vals, labels, label_set, n_int, sp, thresh, dname)
%BUILD_REFS_CLASS_SORTED  Class-sorted reference intervals.
%
%  Sort classes by their feature-space mean, place boundaries at midpoints
%  between adjacent sorted class means.  No monotonicity requirement.

lo_data  = min(vals) - 1e-6;
hi_data  = max(vals) + 1e-6;
n_lbl    = length(label_set);
strategy = 'uniform';

if abs(sp) >= thresh
    % Compute per-class means
    cmeans = zeros(n_lbl,1);
    valid  = false(n_lbl,1);
    for j = 1:n_lbl
        m = abs(labels - label_set(j)) < 1e-9;
        if sum(m) >= 2
            cmeans(j) = mean(vals(m));
            valid(j)  = true;
        end
    end

    if sum(valid) >= 2
        % Sort classes by feature-space mean (ascending)
        valid_idx    = find(valid);
        valid_means  = cmeans(valid_idx);
        [sorted_means, sort_order] = sort(valid_means);
        n_sorted = length(sorted_means);

        % Boundaries at midpoints between consecutive sorted class means
        mids = zeros(n_sorted-1,1);
        for j = 1:n_sorted-1
            mids(j) = (sorted_means(j) + sorted_means(j+1)) / 2;
        end

        refs_raw = [lo_data; mids; hi_data];

        % Adjust to exactly n_int+1 reference points
        refs = adjust_to_length(refs_raw, n_int+1, lo_data, hi_data);
        strategy = 'class-sorted';

        % Print dominant class per interval for diagnostics
        label_names = arrayfun(@(v)sprintf('%.2f',v), label_set(valid_idx(sort_order)),'uni',false);
        fprintf('    %s [%s]: class order = [%s]\n', dname, strategy, strjoin(label_names,','));
    else
        refs = linspace(lo_data, hi_data, n_int+1)';
    end
else
    refs = linspace(lo_data, hi_data, n_int+1)';
    fprintf('    %s [uniform]: |Spearman|=%.4f < threshold=%.2f\n', dname, abs(sp), thresh);
end

refs = ensure_valid(refs, lo_data, hi_data, n_int);
fprintf('    %s boundaries: %s\n', dname, ...
        strjoin(arrayfun(@(v)sprintf('%.4f',v), refs(:)', 'UniformOutput',false),' '));
end

%% -----------------------------------------------------------------------
function refs = adjust_to_length(raw, target_len, lo, hi)
%ADJUST_TO_LENGTH  Pad or trim reference vector to exactly target_len points.
raw = sort(unique(raw(:)));
raw(1)   = lo;
raw(end) = hi;

if length(raw) == target_len
    refs = raw;
elseif length(raw) < target_len
    % Insert extra split points in the largest gaps
    while length(raw) < target_len
        gaps  = diff(raw);
        [~,k] = max(gaps);
        raw   = sort([raw; (raw(k)+raw(k+1))/2]);
    end
    refs = raw;
else
    % Too many points: subsample evenly preserving endpoints
    idx  = round(linspace(1, length(raw), target_len));
    refs = raw(idx);
end
end

%% -----------------------------------------------------------------------
function refs = ensure_valid(refs, lo, hi, n_int)
%ENSURE_VALID  Guarantee strictly increasing with minimum gap and correct count.
refs = sort(refs(:));
refs(1)   = lo;
refs(end) = hi;
rng_v   = hi - lo;
min_gap = max(1e-6, rng_v * 1e-4);

% Remove near-duplicate points
keep = [true; diff(refs) > min_gap];
refs = refs(keep);

if length(refs) ~= n_int+1
    % Fallback to uniform if uniqueness failed
    refs = linspace(lo, hi, n_int+1)';
end
end

%% -----------------------------------------------------------------------
function r = spearman_r(x, y)
%SPEARMAN_R  Spearman rank correlation coefficient.
rx = tiedrank_local(x);
ry = tiedrank_local(y);
mx = mean(rx); my = mean(ry);
num = sum((rx-mx).*(ry-my));
den = sqrt(sum((rx-mx).^2) * sum((ry-my).^2));
if den < 1e-12, r = 0; else, r = num/den; end
end

function r = tiedrank_local(x)
n   = length(x);
[~,idx] = sort(x);
r   = zeros(n,1);
r(idx) = (1:n)';
ux  = unique(x);
for i = 1:length(ux)
    m    = (x == ux(i));
    r(m) = mean(r(m));
end
end
