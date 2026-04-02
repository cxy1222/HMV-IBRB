function f = fun_interval(x)
%FUN_INTERVAL  Training MSE objective for Interval BRB (called by cmaes).
%  Globals: L, N, TrainData, Y1_REF, Y2_REF
global L N TrainData Y1_REF Y2_REF
[y_out,~,~] = interval_inference(x, TrainData, L, N, Y1_REF, Y2_REF);
f = mean((y_out - TrainData(:,3)).^2);
end
