function showpartclusters(pos,idx)

globals;

clusterdir = [visualdir 'cluster/part%d/'];
partclusterimg = [clusterdir 'mix%d.jpg'];
for p = 1:length(idx)
  if exist(sprintf(clusterdir,p),'dir')
    rmdir(sprintf(clusterdir,p),'s');
  end
  mkdir(sprintf(clusterdir,p));
end

patchall = zeros(40,40,3,length(pos),length(idx));
for n = 1:length(pos)
  im = imread(pos(n).im);
  if size(im, 3) == 1
    im = repmat(im,[1 1 3]);
  end
  for p = 1:length(idx)
    x1 = round(pos(n).x1(p));
    y1 = round(pos(n).y1(p));
    x2 = round(pos(n).x2(p));
    y2 = round(pos(n).y2(p));
    patch = subarray(im,y1,y2,x1,x2,0);
    patch = imresize(patch, [40 40]);
    patchall(:,:,:,n,p) = patch;
  end
end

for p = 1:length(idx)
  for m = 1:max(idx{p})
    figure(2); clf; montage(uint8(patchall(:,:,:,idx{p} == m,p)));
    saveas(gcf,sprintf(partclusterimg,p,m));
  end
end