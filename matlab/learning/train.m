function model = train(name, model, pos, neg, warp, iter, C, wpos, maxsize, overlap) 
% model = train(name, model, pos, neg, warp, iter, C, Jpos, maxsize, overlap)
%               1,    2,     3,   4,   5,    6,    7, 8,    9,       10
% Train a structured SVM with latent assignement of positive variables
% pos  = list of positive images with part annotations
% neg  = list of negative images
% warp = 1 uses warped positives
% warp = 0 uses latent positives
% iter is the number of training iterations
%   C  = scale factor for slack loss
% wpos =  amount to weight errors on positives
% maxsize = maximum size of the training data cache (in GB)
% overlap =  minimum overlap in latent positive search

if nargin < 6
  iter = 1;
end

if nargin < 7
  C = 0.002;
end

if nargin < 8
  wpos = 2;
end

if nargin < 9
  % Estimated #sv = (wpos + 1) * # of positive examples
  % maxsize*1e9/(4*model.len)  = # of examples we can store, encoded as 4-byte floats
  no_sv = (wpos+1) * length(pos);
  maxsize = 10 * no_sv * 4 * sparselen(model) / 1e9;
  maxsize = min(max(maxsize,6),7.5);
  % the current version of octave does really bad memory reallocations
  % when slicing, so we can't use too much memory otherwise it starts paging
  if isoctave() maxsize = 1.8; end
end

fprintflush('Using %.1f GB\n',maxsize);

if nargin < 10
  overlap = 0.6;
end

% Vectorize the model
len  = sparselen(model);
nmax = round(maxsize*.25e9/len);

rand('state',0);
globals;

% Define global QP problem
clear global qp;
global qp;
% qp.x(:,i) = examples
% qp.i(:,i) = id
% qp.b(:,i) = bias of linear constraint
% qp.d(i)   = ||qp.x(:,i)||^2
% qp.a(i)   = ith dual variable
qp.x   = zeros(len,nmax,'single');
qp.i   = zeros(5,nmax,'int32');
qp.b   = ones(nmax,1,'single');
qp.d   = zeros(nmax,1,'double');
qp.a   = zeros(nmax,1,'double');
qp.sv  = logical(zeros(1,nmax));  
qp.n   = 0;
qp.lb = [];

[qp.w,qp.wreg,qp.w0,qp.noneg] = model2vec(model);
qp.Cpos = C*wpos;
qp.Cneg = C;
qp.w    = (qp.w - qp.w0).*qp.wreg;

for t = 1:iter,
  fprintflush('\niter: %d/%d\n', t, iter);
  qp.n = 0;
  if warp > 0
    numpositives = poswarp(name, t, model, pos);
  else
    numpositives = poslatent(name, t, model, pos, overlap);
  end
  
  for i = 1:length(numpositives),
    fprintflush('component %d got %d positives\n', i, numpositives(i));
  end
  assert(qp.n <= nmax);
  
  % Fix positive examples as permenant support vectors
  % Initialize QP soln to a valid weight vector
  % Update QP with coordinate descent
  qp.svfix = 1:qp.n;
  qp.sv(qp.svfix) = 1;
  qp_prune();
  qp_opt();
  model = vec2model(qp_w,model);
	interval0 = model.interval;
  model.interval = 2;

  % grab negative examples from negative images
  for i = 1:length(neg)
    fprintflush('\n Image(%d/%d)',i,length(neg));
    im  = imread(neg(i).im);
    [box,model] = detect(im, model, -1, [], 0, i, -1);
    fprintflush(' #cache+%d=%d/%d, #sv=%d, #sv>0=%d, (est)UB=%.4f, LB=%.4f',size(box,1),qp.n,nmax,sum(qp.sv),sum(qp.a>0),qp.ub,qp.lb);
    % Stop if cache is full
    if sum(qp.sv) == nmax,
      break;
    end
  end

  % One final pass of optimization
  qp_opt();
  model = vec2model(qp_w,model);

  fprintflush('\nDONE iter: %d/%d #sv=%d/%d, LB=%.4f\n',t,iter,sum(qp.sv),nmax,qp.lb);

  % Compute minimum score on positive example (with raw, unscaled features)
  r = sort(qp_scorepos());
  model.thresh   = r(ceil(length(r)*.05));
  model.lb = qp.lb;
  model.ub = qp.ub;
  model.interval = interval0;
  % visualizemodel(model);
  % cache model
  % save([cachedir name '_model_' num2str(t)], 'model');
end
fprintflush('qp.x size = [%d %d]\n',size(qp.x));
clear global qp;

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function numpositives = poswarp(name, t, model, pos)

numpos = length(pos);
warped = warppos(name, model, pos);
minsize = prod(model.maxsize*model.sbin);

for i = 1:numpos
	fprintflush('%s: iter %d: warped positive: %d/%d\n', name, t, i, numpos);
	bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
	% skip small examples
	if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
		continue;
	end    
	% get example
	im = warped{i};
	feat = features(im, model.sbin);
	qp_poswrite(feat,i,model);
end
global qp;
numpositives = qp.n;
  
function qp_poswrite(feat,id,model)

len = numel(feat);
ex.id     = [1 id 0 0 0]';
ex.blocks = [];
ex.blocks(end+1).i = model.bias.i;
ex.blocks(end).x   = 1;
ex.blocks(end+1).i = model.filters.i;
ex.blocks(end).x   = feat;
qp_write(ex);
  

% get positive examples using latent detections
% we create virtual examples by flipping each image left to right
function numpositives = poslatent(name, t, model, pos, overlap)
  
numpos = length(pos);
numpositives = zeros(length(model.components), 1);
minsize = prod(model.maxsize*model.sbin);
  
for i = 1:numpos
	fprintflush('%s: iter %d: latent positive: %d/%d\n', name, t, i, numpos);
	% skip small examples
	bbox.xy = [pos(i).x1' pos(i).y1' pos(i).x2' pos(i).y2'];
  if isfield(pos,'mix')
    bbox.m = pos(i).mix;
  end
	area = (bbox.xy(:,3)-bbox.xy(:,1)+1).*(bbox.xy(:,4)-bbox.xy(:,2)+1);
	if any(area < minsize)
		continue;
	end
	
	% get example
	im = imread(pos(i).im);
	[im, bbox] = croppos(im, bbox);
	box = detect(im, model, 0, bbox, overlap, i, 1);
	if ~isempty(box),
		fprintflush(' (comp=%d,sc=%.3f)\n',box(1,end-1),box(1,end));
		c = box(1,end-1);
		numpositives(c) = numpositives(c)+1;
	end
end

% Compute score (weights*x) on positives examples (see qp_write.m)
% Standardized QP stores w*x' where w = (weights-w0)*r, x' = c_i*(x/r)
% (w/r + w0)*(x'*r/c_i) = (v + w0*r)*x'/ C
function scores = qp_scorepos

global qp;
y = qp.i(1,1:qp.n);
I = find(y == 1);
w = qp.w + qp.w0.*qp.wreg;
scores = score(w,qp.x,I) / qp.Cpos;

% Computes expected number of nonzeros in sparse feature vector 
function len = sparselen(model)

numblocks = 0;
for c = 1:length(model.components)
	feat = zeros(model.len,1);
	for p = model.components{c},
		if ~isempty(p.biasid)
			x = model.bias(p.biasid(1));
			i1 = x.i;
			i2 = i1 + numel(x.w) - 1;
			feat(i1:i2) = 1;
			numblocks = numblocks + 1;
		end
		if ~isempty(p.filterid)
			x  = model.filters(p.filterid(1));
			i1 = x.i;
			i2 = i1 + numel(x.w) - 1;
			feat(i1:i2) = 1;
			numblocks = numblocks + 1;
		end
		if ~isempty(p.defid)
			x  = model.defs(p.defid(1));
			i1 = x.i;
			i2 = i1 + numel(x.w) - 1;
			feat(i1:i2) = 1;
			numblocks = numblocks + 1;
		end
	end
	
	% Number of entries needed to encode a block-sparse representation
	%   1 + numberofblocks*2 + #nonzeronumbers
	len = 1 + numblocks*2 + sum(feat);
end
