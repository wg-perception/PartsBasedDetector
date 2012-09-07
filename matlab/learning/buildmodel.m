function jointmodel = buildmodel(name,model,def,idx,K,pa)
% jointmodel = buildmodel(name,model,pa,def,idx,K)
% This function merges together separate part models into a tree structure

globals;

jointmodel.bias    = struct('w',{},'i',{});
jointmodel.filters = struct('w',{},'i',{});
jointmodel.defs    = struct('w',{},'i',{},'anchor',{});
jointmodel.components{1} = struct('biasid',{},'filterid',{},'defid',{},'parent',{});

jointmodel.pa = pa;
jointmodel.maxsize  = model.maxsize;
jointmodel.interval = model.interval;
jointmodel.sbin = model.sbin;
jointmodel.len = 0;

% add children
for i = 1:length(pa)
  child = i;
  parent = pa(child);
  assert(parent < child);
  
  cls = [name '_part_' num2str(child) '_mix_' num2str(K(i))];
  load([cachedir cls]);

  % add bias
  p.biasid = [];
  if parent == 0
    nb  = length(jointmodel.bias);
    b.w = 0;
    b.i = jointmodel.len + 1;
    jointmodel.bias(nb+1) = b;
    jointmodel.len = jointmodel.len + numel(b.w);
    p.biasid = nb+1;
  else
    for k = 1:max(idx{child})
      for l = 1:max(idx{parent})
        % if any(idx{child} == k & idx{parent} == l),
        nb = length(jointmodel.bias);
        b.w = 0;
        b.i = jointmodel.len + 1;
        jointmodel.bias(nb+1) = b;
        jointmodel.len = jointmodel.len + numel(b.w);
        p.biasid(l,k) = nb+1;
      end
    end
  end

  % add filter
  p.filterid = [];
  for k = 1:max(idx{child})    
    nf  = length(jointmodel.filters);
    f.w = model.filters(k).w;
    f.i = jointmodel.len + 1;
    jointmodel.filters(nf+1) = f;
    jointmodel.len = jointmodel.len + numel(f.w);
    p.filterid = [p.filterid nf+1];
  end

  % add deformation parameter
  p.defid = [];
  if parent > 0
    for k = 1:max(idx{child})
      nd  = length(jointmodel.defs);
      d.w = [0.01 0 0.01 0];
      d.i = jointmodel.len + 1;
      x = mean(def{child}(idx{child}==k,1) - def{parent}(idx{child}==k,1)); 
      y = mean(def{child}(idx{child}==k,2) - def{parent}(idx{child}==k,2));
      d.anchor = round([x+1 y+1 0]);
      jointmodel.defs(nd+1) = d;
      jointmodel.len = jointmodel.len + numel(d.w);	
      p.defid = [p.defid nd+1];
    end
  end

  p.parent = parent;
  np = length(jointmodel.components{1});
  jointmodel.components{1}(np+1) = p;
end
