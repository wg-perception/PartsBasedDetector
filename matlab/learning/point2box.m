function pos = point2box(pos,pa)

len = zeros(length(pos),length(pa)-1);
for n = 1:length(pos)
  points = pos(n).point;
  for p = 2:size(points,1)
    len(n,p-1) = norm(abs(points(p,1:2)-points(pa(p),1:2)));
  end
end

r = zeros(1,length(pa)-1);
for i = 1:length(pa)-1
  ratio = log(len(:,i))-log(len(:,1));
  r(i) = exp(median(ratio));
end

boxsize = zeros(1,length(pos));
for n = 1:length(pos)
  ratio = len(n,:)./r;
  boxsize(n) = quantile(ratio,0.75);
end

for n = 1:length(pos)
  pos(n).x1 = pos(n).point(:,1) - boxsize(n)/2;
  pos(n).y1 = pos(n).point(:,2) - boxsize(n)/2;
  pos(n).x2 = pos(n).point(:,1) + boxsize(n)/2;
  pos(n).y2 = pos(n).point(:,2) + boxsize(n)/2;
end