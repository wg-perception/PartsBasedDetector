function boxes = testmodel(name,model,test,suffix)
% boxes = testmodel(name,model,test,suffix)
% Returns candidate bounding boxes after non-maximum suppression

globals;

try
  load([cachedir name '_boxes_' suffix]);
catch
  boxes = cell(1,length(test));
  for i = 1:length(test)
    fprintf([name ': testing: %d/%d\n'],i,length(test));
    im = imread(test(i).im);
    box = detect_fast(im,model,model.thresh);
    boxes{i} = nms(box,0.3);
  end

  if nargin < 4
    suffix = [];
  end
  save([cachedir name '_boxes_' suffix], 'boxes','model');
end
