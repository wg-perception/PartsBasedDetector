function showskeletons(im, boxes, partcolor, parent)

imagesc(im); axis image; axis off; hold on;
if ~isempty(boxes)
  numparts = length(partcolor);
  box = boxes(:,1:4*numparts);
  xy = reshape(box,size(box,1),4,numparts);
  xy = permute(xy,[1 3 2]);
  for n = 1:size(xy,1)
    x1 = xy(n,:,1); y1 = xy(n,:,2); x2 = xy(n,:,3); y2 = xy(n,:,4);
    x = (x1+x2)/2; y = (y1+y2)/2;
    for child = 2:numparts
      x1 = x(parent(child));
      y1 = y(parent(child));
      x2 = x(child);
      y2 = y(child);
      line([x1 x2],[y1 y2],'color',partcolor{child},'linewidth',4);
    end
  end
end
drawnow;
