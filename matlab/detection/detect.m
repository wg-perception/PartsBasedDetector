function [boxes,model,ex] = detect(im, model, thresh, bbox, overlap, id, label)
%        [boxes,model,ex] = detect(im, model, thresh, bbox, overlap, id, label)
% Detect objects in image using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part
%
% If bbox is not empty, we pick best detection with significant overlap. 
% If label is included, we write feature vectors to a global QP structure
%
% This function updates the model (by running the QP solver) if upper and lower bound differs
  
INF = 1e10;

if nargin > 3 && ~isempty(bbox)
  latent = true;
  thresh = -INF;
else
  latent = false;
end

% Compute the feature pyramid and prepare filter
pyra = featpyramid(im,model);
interval = model.interval;
levels = 1:length(pyra.feat);

% Define global QP if we are writing features
% Randomize order to increase effectiveness of model updating
write = false;
if nargin > 5
  global qp;
  write  = true;
	levels = levels(randperm(length(levels)));
end
if nargin < 6
  id = 0;
end
if nargin < 7
  label = 0;
end

% Cache various statistics derived from model
[components,filters,resp] = modelcomponents(model,pyra);
boxes     = zeros(100000,length(components{1})*4+2);
cnt = 0;

ex.blocks = [];
ex.id = [label id 0 0 0];

% Iterate over random permutation of scales and components,
for rlevel = levels
  % Iterate through mixture components
  for c  = randperm(length(model.components))
    parts = components{c};
    numparts = length(parts);

    % Skip if there is no overlap of root filter with bbox
    if latent
      skipflag = 0;
      for k = 1:numparts
				% because all mixtures for one part is the same size, we only need to do this once
				ovmask = testoverlap(parts(k).sizx(1),parts(k).sizy(1),pyra,rlevel,bbox.xy(k,:),overlap);
				if ~any(ovmask)
					skipflag = 1;
					break;
				end
			end
      if skipflag == 1
        continue;
      end
    end

    % Local scores
    for k = 1:numparts
      f = parts(k).filterid;
      level = rlevel-parts(k).scale*interval;
      if isempty(resp{level})
        resp{level} = fconv(pyra.feat{level},filters,1,length(filters));
      end
      for fi = 1:length(f)
        parts(k).score(:,:,fi) = resp{level}{f(fi)};
      end
      parts(k).level = level;
	  
      if latent
				for fi = 1:length(f)
					if isfield(bbox,'m')
						if fi ~= bbox.m(k)
							parts(k).score(:,:,fi) = -INF;
						end
					else
						ovmask = testoverlap(parts(k).sizx(fi),parts(k).sizy(fi),pyra,rlevel,bbox.xy(k,:),overlap);
						tmpscore = parts(k).score(:,:,fi);
						tmpscore(~ovmask) = -INF;
						parts(k).score(:,:,fi) = tmpscore;
					end
				end
			end
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for k = numparts:-1:2
      par = parts(k).parent;
      [msg,parts(k).Ix,parts(k).Iy,parts(k).Im] = passmsg(parts(k),parts(par));
      parts(par).score = parts(par).score + msg;
    end

    % Add bias to root score
    parts(1).score = parts(1).score + parts(1).b;
		[rscore Im] = max(parts(1).score,[],3);
    
    % Zero-out invalid regions in latent mode
    if latent
      thresh = max(thresh,max(rscore(:)));
    end

    [Y,X] = find(rscore >= thresh);
    % Walk back down tree following pointers
    % (DEBUG) Assert extracted feature re-produces score
    for i = 1:length(X)
			cnt = cnt + 1;
      x = X(i);
      y = Y(i);
      m = Im(y,x);
      %[boxes(cnt).xy,boxes(cnt).m,boxes(cnt).v,ex] = backtrack(x,y,m,parts,pyra,ex,write);
			%boxes(cnt).c = c;
			%boxes(cnt).s = rscore(y,x);
      [box,ex] = backtrack(x,y,m,parts,pyra,ex,write);
			boxes(cnt,:) = [box c rscore(y,x)];
      if write && ~latent
        qp_write(ex);
        qp.ub = qp.ub + qp.Cneg*max(1+rscore(y,x),0);
      end
    end
    
    % Crucial DEBUG assertion:
    % If we're computing features, assert extracted feature re-produces score
    % (see qp_writ.m for computing original score)
    if write && ~latent && ~isempty(X) && qp.n < length(qp.a)
      w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
      assert((score(w,qp.x,qp.n) - rscore(y,x)) < 1e-5);   
    end
      
    % Optimize qp with coordinate descent, and update model
     if write && ~latent && ...
            (qp.lb < 0 || 1 - qp.lb/qp.ub > .05 || qp.n == length(qp.sv))
      model = optimize(model);
      [components,filters,resp] = modelcomponents(model,pyra);    
    end
  end
end

%boxes = boxes(1:cnt);
boxes = boxes(1:cnt,:);

if latent && ~isempty(boxes)
  %boxes = boxes(end);
	boxes = boxes(end,:);
  if write
    qp_write(ex);
  end
end


% ----------------------------------------------------------------------
% Helper functions for detection, feature extraction, and model updating
% ----------------------------------------------------------------------

% Cache various statistics from the model data structure for later use  
function [components,filters,resp] = modelcomponents(model,pyra)

components = cell(length(model.components),1);
for c = 1:length(model.components)
	for k = 1:length(model.components{c})
		p = model.components{c}(k);
		[p.sizy,p.sizx,p.w,p.biasI,p.filterI,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
		[p.scale,p.level,p.Ix,p.Iy] = deal(0);

		% store the scale of each part relative to the component root
		par = p.parent;      
		assert(par < k);
		p.b = [model.bias(p.biasid).w];
		p.b = reshape(p.b,[1 size(p.biasid)]);
		p.biasI = [model.bias(p.biasid).i];
		p.biasI = reshape(p.biasI,size(p.biasid));
		
		for f = 1:length(p.filterid)
			x = model.filters(p.filterid(f));
			[p.sizy(f) p.sizx(f) foo] = size(x.w);
			p.filterI(f) = x.i;
		end
		
		for f = 1:length(p.defid)	  
			x = model.defs(p.defid(f));
			p.w(:,f)  = x.w';
			p.defI(f) = x.i;
			ax = x.anchor(1);
			ay = x.anchor(2);    
			ds = x.anchor(3);
			p.scale = ds + components{c}(par).scale;
			% amount of (virtual) padding to hallucinate
			step = 2^ds;
			virtpady = (step-1)*pyra.pady;
			virtpadx = (step-1)*pyra.padx;
			% starting points (simulates additional padding at finer scales)
			p.starty(f) = ay-virtpady;
			p.startx(f) = ax-virtpadx;      
			p.step   = step;
		end
		components{c}(k) = p;
	end
end

resp    = cell(length(pyra.feat),1);
filters = cell(length(model.filters),1);
for i = 1:length(filters)
	filters{i} = model.filters(i).w;
end


% Given a 2D array of filter scores 'child',
% (1) Apply distance transform
% (2) Shift by anchor position of part wrt parent
% (3) Downsample if necessary
function [score,Ix,Iy,Im] = passmsg(child,parent)

K   = length(child.filterid);
Ny  = size(parent.score,1);
Nx  = size(parent.score,2);  
Ix0 = zeros([Ny Nx K]);
Iy0 = zeros([Ny Nx K]);
[Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));

for k = 1:K
	[score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k),child.startx(k),child.starty(k),Nx,Ny,child.step);
end

% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
L  = length(parent.filterid);
N  = Nx*Ny;
i0 = reshape(1:N,Ny,Nx);
[score,Ix,Iy,Im] = deal(zeros(Ny,Nx,L));
for l = 1:L
	b = child.b(1,l,:);
	[score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
	i = i0 + N*(I-1);
	Ix(:,:,l)    = Ix0(i);
	Iy(:,:,l)    = Iy0(i);
	Im(:,:,l)    = I;
end

% Backtrack through dynamic programming messages to estimate part locations
% and the associated feature vector  
function [box,ex] = backtrack(x,y,mix,parts,pyra,ex,write)

numparts = length(parts);
ptr = zeros(numparts,3);
box = zeros(numparts,4);
k   = 1;
p   = parts(k);
ptr(k,:) = [x y mix];
scale = pyra.scale(p.level);
x1  = (x - 1 - pyra.padx)*scale+1;
y1  = (y - 1 - pyra.pady)*scale+1;
x2  = x1 + p.sizx(mix)*scale - 1;
y2  = y1 + p.sizy(mix)*scale - 1;
box(k,:) = [x1 y1 x2 y2];

if write
	ex.id(3:5) = [p.level round(x+p.sizx(mix)/2) round(y+p.sizy(mix)/2)];
	ex.blocks = [];
	ex.blocks(end+1).i = p.biasI;
	ex.blocks(end).x   = 1;
	f  = pyra.feat{p.level}(y:y+p.sizy(mix)-1,x:x+p.sizx(mix)-1,:);
	ex.blocks(end+1).i = p.filterI(mix);
	ex.blocks(end).x   = f;
end
for k = 2:numparts
	p   = parts(k);
	par = p.parent;
	x   = ptr(par,1);
	y   = ptr(par,2);
	mix = ptr(par,3);
	ptr(k,1) = p.Ix(y,x,mix);
	ptr(k,2) = p.Iy(y,x,mix);
	ptr(k,3) = p.Im(y,x,mix);
	scale = pyra.scale(p.level);
	x1  = (ptr(k,1) - 1 - pyra.padx)*scale+1;
	y1  = (ptr(k,2) - 1 - pyra.pady)*scale+1;
	x2  = x1 + p.sizx(ptr(k,3))*scale - 1;
	y2  = y1 + p.sizy(ptr(k,3))*scale - 1;
	box(k,:) = [x1 y1 x2 y2];
	
	if write
		ex.blocks(end+1).i = p.biasI(mix,ptr(k,3));
		ex.blocks(end).x   = 1;
		ex.blocks(end+1).i = p.defI(ptr(k,3));
		ex.blocks(end).x   = defvector(x,y,ptr(k,1),ptr(k,2),ptr(k,3),p);
		x   = ptr(k,1);
		y   = ptr(k,2);
		mix = ptr(k,3);
		f   = pyra.feat{p.level}(y:y+p.sizy(mix)-1,x:x+p.sizx(mix)-1,:);
		ex.blocks(end+1).i = p.filterI(mix);
		ex.blocks(end).x = f;
	end
end
box = reshape(box',1,4*numparts);


% Update QP with coordinate descent
% and return the asociated model
function model = optimize(model)

global qp;
fprintflush('.');  
if qp.lb < 0 || qp.n == length(qp.a),
	qp_opt();
	qp_prune();
else
	qp_one();
end
model = vec2model(qp_w(),model);
 

% Compute the deformation feature given parent locations, 
% child locations, and the child part
function res = defvector(px,py,x,y,mix,part)

probex = ( (px-1)*part.step + part.startx(mix) );
probey = ( (py-1)*part.step + part.starty(mix) );
dx = probex - x;
dy = probey - y;
res = -[dx^2 dx dy^2 dy]';


% Compute a mask of filter reponse locations (for a filter of size sizy,sizx)
% that sufficiently overlap a ground-truth bounding box (bbox) 
% at a particular level in a feature pyramid
function ov = testoverlap(sizx,sizy,pyra,level,bbox,overlap)

scale = pyra.scale(level);
padx  = pyra.padx;
pady  = pyra.pady;
[dimy,dimx,foo] = size(pyra.feat{level});

bx1 = bbox(1);
by1 = bbox(2);
bx2 = bbox(3);
by2 = bbox(4);

% Index windows evaluated by filter (in image coordinates)
x1 = ((1:dimx-sizx+1) - padx - 1)*scale + 1;
y1 = ((1:dimy-sizy+1) - pady - 1)*scale + 1;
x2 = x1 + sizx*scale - 1;
y2 = y1 + sizy*scale - 1;

% Compute intersection with bbox
xx1 = max(x1,bx1);
xx2 = min(x2,bx2);
yy1 = max(y1,by1);
yy2 = min(y2,by2);
w = xx2 - xx1 + 1;
h = yy2 - yy1 + 1;
w(w<0) = 0;
h(h<0) = 0;
inter  = h'*w;

% area of (possibly clipped) detection windows and original bbox
area = (y2-y1+1)'*(x2-x1+1);
box = (by2-by1+1)*(bx2-bx1+1);

% thresholded overlap
ov = inter ./ (area + box - inter) > overlap;
