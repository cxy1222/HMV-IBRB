function bestx=cmaes(x,G,Aeq,beq,ub,lb)

% global L
% global M
% global N
% (mu/mu_w, lambda)-CMA-ES
% CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
% nonlinear function minimization. To be used under the terms of the
% GNU General Public License (http://www.gnu.org/copyleft/gpl.html).
% Copyright: Nikolaus Hansen, 2003-09.
% e-mail: hansen[at]lri.fr
%
% This code is an excerpt from cmaes.m and implements the key parts
% of the algorithm. It is intendend to be used for READING and
% UNDERSTANDING the basic flow and all details of the CMA-ES
% *algorithm*. Use the cmaes.m code to run serious simulations: it
% is longer, but offers restarts, far better termination options,
% and supposedly quite useful output.
%
% URL: http://www.lri.fr/~hansen/purecmaes.m
% References: See end of file. Last change: October, 21, 2010

% --------------------  Initialization --------------------------------
% User defined input parameters (need to be edited)
%strfitnessfct = 'frosenbrock';  % name of objective/fitness function
%FES=0;
%  x=rand(32,1);%产生范围内的随机数N = 20; 

%  II=[1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,3,2;1,3,2;1,3,2;2,3,1;2,3,1;2,3,1;2,3,1;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,2,3;1,3,2;1,3,2;1,3,2;2,3,1;2,3,1;2,3,1;2,3,1];

N = length(x);               % number of objective variables/problem dimension（维数）
xmean = x;    % objective variables initial point（初始种群）
sigma = 0.5;          % coordinate wise standard deviation (step size)（步长）
stopfitness = 1e-10;  % stop if fitness < stopfitness (minimization)（误差）
stopeval = G;%1e3*N^2;   % stop after stopeval number of function evaluations（最大代数）
xzhongxin=x;%?????????
% Strategy parameter setting: Selection
lambda = 10+floor(3*log(N));  % population size, offspring number（后代的数量）
mu = lambda/2;               % number of parents/points for recombination
weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
mu = floor(mu);
weights = weights/sum(weights);     % normalize recombination weights array
mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

% Strategy parameter setting: Adaptation
cc = (4 + mueff/N) / (N+4 + 2*mueff/N); % time constant for cumulation for C
cs = (mueff+2) / (N+mueff+5);  % t-const for cumulation for sigma control
c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma
% usually close to 1
% Initialize dynamic (internal) strategy parameters and constants
pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
B = eye(N,N);                       % B defines the coordinate system
D = ones(N,1);                      % diagonal D defines the scaling
C = B * diag(D.^2) * B';            % covariance matrix C
invsqrtC = B * diag(D.^-1) * B';    % C^-1/2
eigeneval = 0;                      % track update of B and D
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of
%   ||N(0,I)|| == norm(randn(N,1))
%out.dat = []; out.datx = [];  % for plotting output

% -------------------- Generation Loop --------------------------------
counteval = 0;  % the next 40 lines contain the 20 lines of interesting code
while counteval < stopeval
    
    % Generate and evaluate lambda offspring
    for k=1:lambda
        arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C)

        for kk=1:N
            if arx(kk,k)>ub(kk)
                arx(kk,k)=ub(kk);
            end
            if arx(kk,k)<lb(kk)
                arx(kk,k)=lb(kk);
            end
        end
      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%投影算子%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [aa,b]=size(Aeq);
        a=0;
        for i=1:aa
            if beq(i)==1
                a=a+1;
            end
        end
        num=zeros(1,a);
        for i=1:a
            for j=1:b
                if Aeq(i,j)==1
                    num(i)=num(i)+1;  %记录Aeq中每行1的个数
                    ini(i)=j;
                end
            end
            ini(i)=ini(i)-num(i)+1;   %记录Aeq中每行1的起始列号
        end
        for i=1:a
            A=ones(1,num(i));
            arx(ini(i):(ini(i)+num(i)-1),k)=arx(ini(i):(ini(i)+num(i)-1),k)-A'*inv(A*A')*(A*arx(ini(i):(ini(i)+num(i)-1),k)-1);
            for j=ini(i):(ini(i)+num(i)-1)
                yu=0;
                if arx(j,k)<0
                    arx(j,k)=0;
                    yu=yu+arx(j,k);
                end
            end
            index=0;
            for j=ini(i):(ini(i)+num(i)-1)
                if arx(j,k)~=0
                    index=index+1;
                end
            end
            yu=yu/index;
            for j=ini(i):(ini(i)+num(i)-1)
                if arx(j,k)~=0
                    arx(j,k)=arx(j,k)+yu;
                end
            end
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       for ku=1:40
%                           for n=1:3
%                                BeliefF(ku,n)=arx((ku-1)*3+n,k);      % the optimized BRB
%                           end
%                       end
%                       for ku=1:40
%                            qq(ku,:) = sort(BeliefF(ku,:));
%                            for p = 1:3   
%                                xx(ku,p)=qq(ku,II(ku,p));
%                            end
%                        end
%                       for ku=1:40
%                            for n=1:3
%                                arx((ku-1)*3+n,k)=xx(ku,n);      % the optimized BRB
%                            end
%                       end   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        arfitness(k) = fun_interval(arx(:,k)'); % objective function call
        %FES=FES+1;
    end
    counteval = counteval+1;
    mse1(counteval)=min(arfitness);
    % Sort by fitness and compute weighted mean into xmean
    [arfitness, arindex] = sort(arfitness);  % minimization
    xold = xmean;
    xmean = arx(:,arindex(1:mu)) * weights;  % recombination, new mean value
    juli(counteval,1)=pdist([xzhongxin';xmean'],'euclidean');
    % Cumulation: Update evolution paths
    ps = (1-cs) * ps ...
        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
    hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/lambda))/N < 2 + 4/(N+1);
    pc = (1-cc) * pc ...
        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
    
    % Adapt covariance matrix C
    artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu));  % mu difference vectors
    C = (1-c1-cmu) * C ...                   % regard old matrix
        + c1 * (pc * pc' ...                % plus rank one update
        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update
    
    % Adapt step size sigma
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));
    %if sigma<10e-4
    %       sigma=0.5;
    %end
    % Update B and D from C
    if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
        eigeneval = counteval;
        C = triu(C) + triu(C,1)'; % enforce symmetry
        [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
        D = sqrt(diag(D));        % D contains standard deviations now
        invsqrtC = B * diag(D.^-1) * B';
    end
    % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
    if counteval==1
        bestf=arfitness(1);
        bestx=arx(:,arindex(1));
    else
        if arfitness(1)<bestf
            bestf=arfitness(1);
            bestx=arx(:,arindex(1));
        end
    end
    % if arfitness(1) <= stopfitness || max(D) > 1e7 * min(D)
    %   break;
    % end
    counteval
end % while, end generation loop
%   hold on;
% for i=1:counteval
%     
%     llkk(i)=i;
% end
%   plot(llkk,mse1,'g');
%   hold on;
end


