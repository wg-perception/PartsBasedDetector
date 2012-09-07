function deffeat = data_def(pos,model)
% get absolute positions of parts with respect to HOG cell

width  = zeros(1,length(pos));
height = zeros(1,length(pos));
points = zeros(size(pos(1).point,1),size(pos(1).point,2),length(pos));
for n = 1:length(pos)
  width(n)  = pos(n).x2(1) - pos(n).x1(1) + 1;
  height(n) = pos(n).y2(1) - pos(n).y1(1) + 1;
  points(:,:,n) = pos(n).point;
end
scale = sqrt(width.*height)/sqrt(model.maxsize(1)*model.maxsize(2));
scale = [scale; scale];

deffeat = cell(1,size(points,1));
for p = 1:size(points,1)
  def = squeeze(points(p,1:2,:));
  deffeat{p} = (def ./ scale)';
end