function [y] = simvardgp(T,N,L,PHI,PSI)
%--------------------------------------------------------------------------
%   PURPOSE:
%      Get matrix of Y generated from a VAR model
%--------------------------------------------------------------------------
%   INPUTS:
%     T     - Number of observations (rows of Y)
%     N     - Number of series (columns of Y)
%     L     - Number of lags
%
%   OUTPUT:
%     y     - [T x N] matrix generated from VAR(L) model
% -------------------------------------------------------------------------

randn('seed',sum(100*clock));
rand('seed',sum(100*clock));
%-----------------------PRELIMINARIES--------------------
if nargin==0;
    T = 50;           %Number of time series observations (T)
    N = 6;             %Number of cross-sectional observations (N)
    L = 1;             %Lag order

    PHI = [1   0   0   0   0   0;
           0   1   0   0   0   0;
           0   0   1   0   0   0;
           0   0   0   1   0   0;
           0   0   0   0   1   0;
           0   0   0   0   0   0;];
       
    PSI = [1 .5 .5 .5 .5 .5 ;
           0  1  0  0  0  0 ;
           0  0  1  0  0  0 ;
           0  0  0  1  0  0 ;
           0  0  0  0  1  0 ;
           0  0  0  0  0  1;];

end
sigma = inv(PSI*PSI');
%---------------------------------------
% % Ask user if a constant is desired
% f = input('Do you want to include a constant? <yes/no>: ','s');
% if strcmp(f,'yes')  % compare strings. If f = 'yes' then
%     const = 1;
% else                % elseif f ='no' then
    %const=0;
% end


%----------------------GENERATE--------------------------
% Set storage in memory for y
% First L rows are created randomly and are used as 
% starting (initial) values 
y =[rand(L,N) ; zeros(T-L,N)];

% Now generate Y from VAR (L,PHI,PSI)
for nn = L:T    
    u = chol(sigma)'*randn(N,1);
    ylag = mlag(y,L);
    y(nn,:) = ylag(nn,:)*PHI + u';
end
% %-------------------------MLE----------------------------
% % Now we can estimate PHI,PSI using OLS. First transform
% % data to get Y and X matrices. X = constant + lagged(Y)
% ylag = mlag(y,L); % Create lagged Y matrix 
% 
% if const==1  % Create X matrix with or without a constant
%     xmat = [ones(T-L,1) ylag(L+1:T,:)];
% else
%     xmat = ylag(L+1:T,:);
% end
% % Chop off first L obs from Y, to match dimensions
% % of X matrix
% ymat = y(L+1:T,:); 
% 
% % Now get MLE quantities.
% PHI_M = inv(xmat'*xmat)*(xmat'*ymat);
% SSE = (ymat - xmat*PHI_M)'*(ymat - xmat*PHI_M);