function [apk prec rec] = eval_apk(ca,gt,thresh)

if nargin < 4
  thresh  = 0.5;
end

[sc,si] = sort(cat(1,ca.score),'descend');
ca = ca(si);

numca = numel(ca);

tp = zeros(numca,1);
fp = zeros(numca,1);
for n = 1:numca
  i = ca(n).fr;
  if gt(i).numgt == 0 % no positive instance in image
    fp(n) = 1; % false positive
    continue;
  end
  % assign detection to ground truth object if any
  dist = sqrt(sum((ca(n).point-gt(i).point).^2,2));
  dist  = dist ./ gt(i).scale;
  
  [distmin jmin] = min(dist);
  % assign detection as true positive/don't care/false positive
  if distmin <= thresh
    if ~gt(i).det(jmin)
      tp(n)=1; % true positive
      gt(i).det(jmin)=true;
    else
      fp(n)=1; % false positive (multiple detection)
    end
  else
    fp(n)=1; % false positive
  end
end

fp = cumsum(fp);
tp = cumsum(tp);
rec = tp/sum(cat(1,gt.numgt));
prec = tp./(fp+tp);

apk = VOCap(rec,prec);
