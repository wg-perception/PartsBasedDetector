function visualizemodel(model)

pad = 2;
bs = 20;

% assuming only one component
c = model.components{1};
numparts = length(c);

Nmix = zeros(1,numparts);
for k = 1:numparts
  Nmix(k) = length(c(k).filterid);
end

ovec = [0 1 0 -1; 1 0 -1 0];
I = zeros(numparts,size(ovec,2));
for k = 2:numparts
  part = c(k);
  anchor = zeros(Nmix(k),2);
  for j = 1:Nmix(k) 
    def = model.defs(part.defid(j));
    anchor(j,:) = [def.anchor(1) def.anchor(2)];
  end
  [dummy I(k,:)] = max(anchor * ovec,[],1);
end
    
clf;
for i = 1:size(ovec,2)
  subplot(2,2,i);

  part = c(1);
  % part filter
  w = model.filters(part.filterid(1)).w;
  w = foldHOG(w);
  scale = max(abs(w(:)));
  p = HOGpicture(w, bs);
  p = padarray(p, [pad pad], 0);
  p = uint8(p*(255/scale));    
  % border 
  p(:,1:2*pad) = 128;
  p(:,end-2*pad+1:end) = 128;
  p(1:2*pad,:) = 128;
  p(end-2*pad+1:end,:) = 128;
  im = p;
  startpoint = zeros(numparts,2);
  startpoint(1,:) = [0 0];

  for k = 2:numparts
    part = c(k);
    parent = c(k).parent;

    fi = I(k,i);

    % part filter
    w = model.filters(part.filterid(fi)).w;
    w = foldHOG(w);
    scale = max(abs(w(:)));
    p = HOGpicture(w, bs);
    p = padarray(p, [pad pad], 0);
    p = uint8(p*(255/scale));    
    % border 
    p(:,1:2*pad) = 128;
    p(:,end-2*pad+1:end) = 128;
    p(1:2*pad,:) = 128;
    p(end-2*pad+1:end,:) = 128;

    % paste into root
    def = model.defs(part.defid(fi));

    x1 = (def.anchor(1)-1)*bs+1 + startpoint(parent,1);
    y1 = (def.anchor(2)-1)*bs+1 + startpoint(parent,2);

    [H W] = size(im);
    imnew = zeros(H + max(0,1-y1), W + max(0,1-x1));
    imnew(1+max(0,1-y1):H+max(0,1-y1),1+max(0,1-x1):W+max(0,1-x1)) = im;
    im = imnew;

    startpoint = startpoint + repmat([max(0,1-x1) max(0,1-y1)],[numparts,1]);

    x1 = max(1,x1);
    y1 = max(1,y1);
    x2 = x1 + size(p,2)-1;
    y2 = y1 + size(p,1)-1;

    startpoint(k,1) = x1 - 1;
    startpoint(k,2) = y1 - 1;

    im(y1:y2, x1:x2) = p;
  end

  % plot parts   
  imagesc(im); colormap gray; axis equal; axis off; drawnow;
end