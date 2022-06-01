function r = bcs(X, Y, sigma, beta)
A = cell2mat([X{:}]);
y = cell2mat([Y{:}]);
x = cell2mat(beta)';
A = double(reshape(A, 1024, 500))'; %N*P
y = double(reshape(y, 500, 1));

N = size(A,2); % signal length
%T = 10;  % number of spikes
K = size(A,1); % number of CS measurements
%sigma = 0.05;

% solve by BP
x0 = A'*inv(A*A')*y;
% take epsilon a little bigger than sigma*sqrt(K)
epsilon =  sigma*sqrt(K)*sqrt(1 + 2*sqrt(2)/sqrt(K));                                                                                                              
tic;
x_BP = l1qc_logbarrier(x0, A, [], y, epsilon, 1e-3);
fprintf(1,'BP number of nonzero weights: %d\n',sum(x_BP~=0));
t_BP = toc;
E_BP = norm(x-x_BP)/norm(x);
disp(['BP: ||I_hat-I||/||I|| = ' num2str(E_BP) ', time = ' num2str(t_BP) ' secs']);



initsigma2 = std(y)^2/1e2;
tic;
[weights,used,sigma2,errbars] = BCS_fast_rvm(A,y,initsigma2,1e-8);
t_BCS = toc;
fprintf(1,'BCS number of nonzero weights: %d\n',length(used));
x_BCS = zeros(N,1); err = zeros(N,1);
x_BCS(used) = weights; err(used) = errbars;

E_BCS = norm(x-x_BCS)/norm(x);
disp(['BCS: ||I_hat-I||/||I|| = ' num2str(E_BCS) ', time = ' num2str(t_BCS) ' secs']);

r = [x_BP, x_BCS];

end