function neg = getNegativeData(directory, im_regex)

  % remove trailing slash from the directory if need be
  if isequal(directory(end), '/') directory = directory(1:end-1); end
  
  % get the directory of positive examples
  contents = dir(directory);
  negim    = arrayfun(@(x) regexMatch(x.name, im_regex), contents);
  negim    = contents(logical(negim));
  
  % get the number of examples
  numnegim = length(negim);
  
  % import the examples into the structure
  for n = 1:numnegim
    neg(n).im    = [directory '/' negim(n).name];
  end
end

function in = regexMatch(string, regex)
  if strfind(string, regex), in = 1; else in = 0; end
end