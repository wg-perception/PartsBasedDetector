function boxes = testmodel_gtbox(name,model,test,suffix)
% boxes = testmodel_gtbox(name,model,test,suffix)
% Returns highest scoring pose that sufficiently overlaps a detection window
% 1) Construct ground-truth bounding box
% 2) Compute all candidates that sufficiently overlap it
% 3) Return highest scoring one  

globals;

try
  load([cachedir name '_boxes_gtbox_' suffix]);
catch
  boxes = cell(1,length(test));
  for i = 1:length(test)
    fprintf([name ': testing: %d/%d\n'],i,length(test));
    im = imread(test(i).im);
    box = detect_fast(im,model,model.thresh);
    x = test(i).point(:,1);
    y = test(i).point(:,2);
    gtbox = [min(x) min(y) max(x) max(y)];
    boxes{i} = bestoverlap(box,gtbox,0.3);
  end

  if nargin < 4 
    suffix = [];
  end
  save([cachedir name '_boxes_gtbox_' suffix], 'boxes','model');
end
