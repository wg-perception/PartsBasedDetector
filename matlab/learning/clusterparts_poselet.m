function idx = clusterparts_poselet(deffeat,defvis,K,co)

R = 100;

idx = cell(1,length(deffeat));
for p = 1:length(deffeat)
	% create clustering feature
  X = [];
  for i = 1:length(deffeat)
    if co(p,i) == 1
      X = [X deffeat{i} - deffeat{p}]; 
    end
  end
% 	X = X(defvis(:,p).*defvis(:,co(p))==1,:);
  % try multiple times kmeans
  gInd = cell(1,R);
  cen  = cell(1,R);
  sumdist = zeros(1,R);
  for trial = 1:R
    [gInd{trial} cen{trial} sumdist(trial)] = k_means(X,K(p));
  end
  [dummy ind] = min(sumdist);
	idx{p} = zeros(size(deffeat{p},1),1);
% 	idx{p}(defvis(:,p).*defvis(:,co(p))==1) = gInd{ind(1)};
  idx{p} = gInd{ind(1)};
end