function model = initmodel(pos,sbin,tsize)

% aspect ratio is kept as 1, so no need to pick mode of aspect ratios
aspect = 1;
% all the part templates currently have the same size, so no need to
% compute template size for every part
w = zeros(1,length(pos));
h = zeros(1,length(pos));
for n = 1:length(pos)
  w(n) = pos(n).x2(1) - pos(n).x1(1) + 1;
  h(n) = pos(n).y2(1) - pos(n).y1(1) + 1;
end
% pick 5 percentile area
areas = sort(h.*w);
area = areas(floor(length(areas) * 0.05));
% pick dimensions
nw = sqrt(area/aspect);
nh = nw*aspect;
nf = length(features(zeros([3 3 3]),1));

% size of HOG features
if nargin < 2
  model.sbin = 8;
else
  model.sbin = sbin;
end

% pick dimensions
if nargin < 3
  tsize = [floor(nh/model.sbin) floor(nw/model.sbin) nf];
end

% bias
b.w = 0;
b.i = 1;

% filter
f.w = zeros(tsize);
f.i = 1+1;

% set up one component model
c(1).biasid = 1;
c(1).defid = [];
c(1).filterid = 1;
c(1).parent = 0;
model.bias(1)    = b;
model.defs       = [];
model.filters(1) = f;
model.components{1} = c;

% initialize the rest of the model structure
model.interval = 10;
model.maxsize = tsize(1:2);
model.len = 1+prod(tsize);

