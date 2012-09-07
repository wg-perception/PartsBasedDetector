function box = bestoverlap(boxes,gtbox,overlap)

box = [];
if isempty(boxes) || isempty(gtbox)
  return;
end

x1 = gtbox(1); y1 = gtbox(2); x2 = gtbox(3); y2 = gtbox(4);
area = (x2-x1+1).*(y2-y1+1);

b = boxes(:,1:floor(size(boxes, 2)/4)*4);
b = reshape(b,size(b,1),4,size(b,2)/4);
bx = .5*b(:,1,:) + .5*b(:,3,:);
by = .5*b(:,2,:) + .5*b(:,4,:);
bx1 = min(bx,[],3);
bx2 = max(bx,[],3);
by1 = min(by,[],3);
by2 = max(by,[],3);

xx1 = max(x1,bx1);
yy1 = max(y1,by1);
xx2 = min(x2,bx2);
yy2 = min(y2,by2);

w = xx2-xx1+1; w(w<0) = 0;
h = yy2-yy1+1; h(h<0) = 0;
inter = w.*h;
o = inter / area;
I = find(o > overlap);

if ~isempty(I)
  [val ind] = max(boxes(I,end));
  box  = boxes(I(ind),:);
end