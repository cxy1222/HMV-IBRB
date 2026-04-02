function y_class = classify_output(y_cont, label_set)
%CLASSIFY_OUTPUT  Snap continuous output to nearest valid label.
y_cont    = y_cont(:);
label_set = label_set(:)';
T         = numel(y_cont);
y_class   = zeros(T,1);
for i = 1:T
    [~,k]      = min(abs(y_cont(i) - label_set));
    y_class(i) = label_set(k);
end
end
