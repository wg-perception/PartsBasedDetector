function model = mergemodels(models)
% model = mergemodels(models)
% Merge a set of models into a single mixture model.

model = models{1};
for m = 2:length(models)
  % merge bias
  nb = length(model.bias);
  for i = 1:length(models{m}.bias)
    x = models{m}.bias(i);
    x.i = x.i + model.len;
    model.bias(nb+1) = x;
  end
  
  % merge filters
  nf = length(model.filters);
  for i = 1:length(models{m}.filters)
    x   = models{m}.filters(i);
    x.i = x.i + model.len;
    model.filters(nf+i) = x;
  end
    
  % merge defs
  nd = length(model.defs);
  for i = 1:length(models{m}.defs)
    x   = models{m}.defs(i);
    x.i = x.i + model.len;
    model.defs(nd+i) = x;
  end
  
  % merge components
  nc = length(model.components);
  for i = 1:length(models{m}.components)
    x = models{m}.components{i};
    for j = 1:length(x),
      x(j).biasid   = x(j).biasid   + nb;
      x(j).defid    = x(j).defid    + nd;
      x(j).filterid = x(j).filterid + nf;
    end
    model.components{nc+i} = x;
  end
  
  model.maxsize = max(model.maxsize, models{m}.maxsize);
  model.len = model.len + models{m}.len;
end
