function [gIdx,c,sumdist]=k_means(X,k)
% K_MEANS    k-means clustring
%   IDX = k_means(X, K) partititions the N x P data matrix X into K
%   clusters through a fully vectorized algorithm, where N is the number of
%   data points and P is the number of dimensions (variables). The
%   partition minimizes the sum of point-to-cluster-centroid Euclidean
%   distances of all clusters. The returned N x 1 vector IDX contains the
%   cluster indices of each point.
%
%   IDX = k_means(X, C) works with the initial centroids, C, (K x P).
%
%   [IDX, C] = k_means(X, K) also returns the K cluster centroid locations
%   in the K x P matrix, C.
%
% See also kmeans

% Version 2.0, by Yi Cao at Cranfield University on 27 March 2008.

% Example 1: small data set
%{
N=200;
X = [randn(N,2)+ones(N,2); randn(N,2)-ones(N,2)];
[cidx, ctrs] = k_means(X, 2);
plot(X(cidx==1,1),X(cidx==1,2),'r.',X(cidx==2,1),X(cidx==2,2),'b.', ctrs(:,1),ctrs(:,2),'kx');
%}

% Example 2: large data set
%{
N=20000;
X = [randn(N,2)+ones(N,2); randn(N,2)-ones(N,2)];
tic
[cidx, ctrs] = k_means(X, 2);
toc
plot(X(cidx==1,1),X(cidx==1,2),'r.',X(cidx==2,1),X(cidx==2,2),'b.', ctrs(:,1),ctrs(:,2),'kx');
%}

% Example 3: large data set with 5 centroids 
%{
N=20000;
X = [randn(N,2)+ones(N,2); randn(N,2)-ones(N,2)];
tic
[cidx, ctrs] = k_means(X, 5);
toc
plot(X(cidx==1,1),X(cidx==1,2),'.',...
X(cidx==2,1),X(cidx==2,2),'.',...
X(cidx==3,1),X(cidx==3,2),'.',...
X(cidx==4,1),X(cidx==4,2),'.',...
X(cidx==5,1),X(cidx==5,2),'.',...
ctrs(:,1),ctrs(:,2),'+','linewidth',2)
%}

% Example 4: Comparison with kmeans in Statistics Toolbox
%{
N=20000;
X = [randn(N,2)+ones(N,2); randn(N,2)-ones(N,2)];
rand('state',0);
tic
cidx = k_means(X, 20);
toc
% Compare with kmeans in Statistis Toolbox
rand('state',0);
tic,
cidx1 = kmeans(X, 20, 'Option', statset('MaxIter',200));
toc
%}

% Check input and output
error(nargchk(2,2,nargin));
% error(nargoutchk(0,2,nargout));

[n,m]=size(X);

% Check if second input is centroids
if ~isscalar(k)
    c=k;
    k=size(c,1);
else
    c=X(ceil(rand(k,1)*n),:);
end

% allocating variables
g0=ones(n,1);
gIdx=zeros(n,1);
D=zeros(n,k);

% Main loop converge if previous partition is the same as current
while any(g0~=gIdx)
%     disp(sum(g0~=gIdx))
    g0=gIdx;
    % Loop for each centroid
    for t=1:k
        d=zeros(n,1);
        % Loop for each dimension
        for s=1:m
            d=d+(X(:,s)-c(t,s)).^2;
        end
        D(:,t)=d;
    end
    % Partition data to closest centroids
    [z,gIdx]=min(D,[],2);
    % Update centroids using means of partitions
    for t=1:k
        ct = mean(X(gIdx==t,:));
        if isempty(ct)
            c(t,:) = NaN;
        else
            c(t,:)=ct;
        end
    end
%     for t=1:m
%         c(:,t)=accumarray(gIdx,X(:,t),[],@mean);
%     end
end
sumdist = sum(z);
