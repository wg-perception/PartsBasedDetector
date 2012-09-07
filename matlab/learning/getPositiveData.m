function [pos test] = getPositiveData(directory, im_regex, lm_regex, split)

  % remove trailing slash from the directory if need be
  if isequal(directory(end), '/') directory = directory(1:end-1); end
  
  % get the directory of positive examples
  contents = dir(directory);
  posim    = arrayfun(@(x) regexMatch(x.name, im_regex), contents);
  poslm    = arrayfun(@(x) regexMatch(x.name, lm_regex), contents);
  posim    = contents(logical(posim));
  poslm    = contents(logical(poslm));
  
  % get the number of examples
  numposim = length(posim);
  numposlm = length(poslm);
  if ~isequal(numposim, numposlm) error('The number of matched images and annotations is not equal'); end
  
  % import the examples into the structure
  for n = 1:numposim
    pos(n).im    = [directory '/' posim(n).name];
    pos(n).point = dlmread([directory '/' poslm(n).name]);
  end
  
  % split them for training and testing
  N = randperm(numposim);
  test = pos(N(ceil(numposim*split):end));
  pos  = pos(N(1:floor(numposim*split)));
  
end

function in = regexMatch(string, regex)
  if strfind(string, regex), in = 1; else in = 0; end
end
