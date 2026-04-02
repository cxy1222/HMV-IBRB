function f = funtest_interval(x)
%FUNTEST_INTERVAL  Evaluation MSE; stores predictions in global KK.
%  Globals: L, N, KK, AllData, Y1_REF, Y2_REF
global L N KK AllData Y1_REF Y2_REF
[y_out,~,~] = interval_inference(x, AllData, L, N, Y1_REF, Y2_REF);
KK = y_out(:)';
f  = mean((y_out - AllData(:,3)).^2);
end
