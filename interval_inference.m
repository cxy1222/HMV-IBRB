function [y_out, belief_out, rule_counts] = interval_inference(x, data, L, N, Y1_REF, Y2_REF)
%INTERVAL_INFERENCE  Core Interval BRB inference engine (v3).
%
%  Parameter layout: x = [betal(L x N);  r_vec(L);  w_vec(L)]
%    Rules  1..n_int correspond to Dim1 intervals
%    Rules n_int+1..L correspond to Dim2 intervals
%
%  For each sample: find interval for Dim1 and Dim2,
%  call ER_Rule to fuse two activated rules, output belief distribution
%  and expected-utility prediction.

T     = size(data,1);
n_int = length(Y1_REF) - 1;     % = L/2

Doutput = linspace(0,1,N);      % {0, 0.25, 0.5, 0.75, 1.0}

% --- Unpack parameter vector -----------------------------------------
betal = reshape(x(1:L*N), N, L)';   % L x N
r_vec = x(L*N+1    : L*N+L);
w_vec = x(L*N+L+1  : L*N+L+L);

% Normalise each rule's belief row (safety: CMA-ES may drift off-simplex)
for k = 1:L
    b = betal(k,:);
    b(isnan(b)|b<0) = 0;
    if sum(b) <= 1e-12, b = ones(1,N)/N; end
    betal(k,:) = b / sum(b);
end
r_vec = min(max(real(r_vec), 1e-4), 1-1e-4);
w_vec = min(max(real(w_vec), 1e-4), 1-1e-4);

y_out       = zeros(T,1);
belief_out  = repmat(1/N, T, N);
rule_counts = zeros(L,1);

for i = 1:T
    % Clip to reference range
    v1 = min(max(data(i,1), Y1_REF(1)), Y1_REF(end));
    v2 = min(max(data(i,2), Y2_REF(1)), Y2_REF(end));

    % Find interval: left-closed right-open; last interval fully closed
    inter = find_interval(v1, Y1_REF, n_int);
    num   = find_interval(v2, Y2_REF, n_int);

    r1 = inter;        % Dim1 rule index
    r2 = n_int + num;  % Dim2 rule index

    rule_counts(r1) = rule_counts(r1) + 1;
    rule_counts(r2) = rule_counts(r2) + 1;

    p       = [betal(r1,:); betal(r2,:)];
    r_local = [r_vec(r1);   r_vec(r2)];
    w_local = [w_vec(r1);   w_vec(r2)];

    s      = ER_Rule(r_local, w_local, p);
    belief = s(3:2+N);
    belief = real(belief(:)');
    belief(isnan(belief)|isinf(belief)|belief<0) = 0;
    if sum(belief) <= 1e-12
        belief = ones(1,N)/N;
    else
        belief = belief / sum(belief);
    end

    belief_out(i,:) = belief;
    y_out(i)        = Doutput * belief';
end
end

%% -----------------------------------------------------------------------
function k = find_interval(v, REF, n_int)
% Binary-search style: left-closed, right-open; last interval closed.
k = n_int;  % default = last
for j = 1:n_int-1
    if v >= REF(j) && v < REF(j+1)
        k = j; return;
    end
end
% Last interval: [REF(n_int), REF(n_int+1)]
end
