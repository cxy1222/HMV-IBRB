function x0 = build_init_x0(trainData, L, N, n_int, label_set, ...
                             Y1_REF, Y2_REF, sp1, sp2)
%BUILD_INIT_X0  Spearman-aware CMA-ES warm start.
%
%  Belief degrees: initialised from empirical label histogram per interval.
%  Reliability & weight: set proportional to |Spearman| so ER_Rule
%    down-weights noisy dimensions automatically.
%    r_dim = clip(0.5 + 0.4*|sp|, 0.1, 0.95)
%  This is critical for LDA/PCA where Dim2 is near-noise.

label_set = label_set(:)';
n_lbl     = length(label_set);

x0 = ones(L*N + L + L, 1) * (1/N);   % uniform prior

% --- Belief degrees from label histograms ----------------------------
for inter = 1:n_int
    x0 = set_belief(x0, inter, trainData(:,1), trainData(:,3), ...
                    Y1_REF(inter), Y1_REF(inter+1), inter<n_int, label_set, n_lbl, N);
end
for num = 1:n_int
    rule_k = n_int + num;
    x0 = set_belief(x0, rule_k, trainData(:,2), trainData(:,3), ...
                    Y2_REF(num), Y2_REF(num+1), num<n_int, label_set, n_lbl, N);
end

% --- Reliability & weight: proportional to |Spearman| ---------------
r1 = clip_r(0.5 + 0.4*abs(sp1));
r2 = clip_r(0.5 + 0.4*abs(sp2));

x0(L*N+1        : L*N+n_int)   = r1;   % Dim1 reliability
x0(L*N+n_int+1  : L*N+L)       = r2;   % Dim2 reliability
x0(L*N+L+1      : L*N+L+n_int) = r1;   % Dim1 weight
x0(L*N+L+n_int+1: end)         = r2;   % Dim2 weight

x0 = min(max(x0, 1e-4), 1-1e-4);
end

%% -----------------------------------------------------------------------
function x0 = set_belief(x0, rule_k, dim_vals, labels, lo, hi, right_open, ...
                         label_set, n_lbl, N)
if right_open
    mask = (dim_vals >= lo) & (dim_vals < hi);
else
    mask = (dim_vals >= lo) & (dim_vals <= hi);
end
if sum(mask) < 2, return; end
counts = zeros(n_lbl,1);
for j = 1:n_lbl
    counts(j) = sum(abs(labels(mask) - label_set(j)) < 1e-9);
end
counts = counts + 0.05;           % small Laplace smoothing
b = counts / sum(counts);
b = min(max(b, 1e-4), 1-1e-4);
b = b / sum(b);
% Pad or trim to N output levels
if length(b) > N, b = b(1:N); end
if length(b) < N, b = [b(:); ones(N-length(b),1)/N]; end
b = b / sum(b);
x0((rule_k-1)*N+1 : rule_k*N) = b;
end

function r = clip_r(r)
r = min(max(r, 0.10), 0.95);
end
