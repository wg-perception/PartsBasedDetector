% Set up global paths used throughout the code
addpath learning;
addpath detection;
addpath visualization;
addpath evaluation;
if isoctave()
  addpath oct;
else
  addpath mex;
end

% directory for caching models, intermediate data, and results
cachedir = 'cache/';
if ~exist(cachedir,'dir')
  mkdir(cachedir);
end

if ~exist([cachedir 'imrotate/'],'dir')
  mkdir([cachedir 'imrotate/']);
end

if ~exist([cachedir 'imflip/'],'dir')
  mkdir([cachedir 'imflip/']);
end
