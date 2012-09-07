% Prunes qp to current active constraints (eg., support vectors)
function n = qp_prune
global qp

% if cache is full of support vectors, only keep non-zero (and fixed) ones
if all(qp.sv),
  qp.sv = qp.a > 0;
  qp.sv(qp.svfix) = 1;
end

I = find(qp.sv > 0);
n = length(I);
assert(n > 0);

qp.l = 0;
qp.w = zeros(size(qp.w));
k    = length(qp.w);
for j = 1:n,
  i = I(j);
  qp.x(:,j) = qp.x(:,i);
  qp.i(:,j) = qp.i(:,i);
  qp.b(j)   = qp.b(i);
  qp.d(j)   = qp.d(i);
  qp.a(j)   = qp.a(i);
  qp.sv(j)  = qp.sv(i);
  qp.l = qp.l +   double(qp.b(j))*qp.a(j);
  qp.w = qp.w + sparse2dense(qp.x(:,j),k)*qp.a(j);
end

qp.sv(1:n)     = 1;
qp.sv(n+1:end) = 0;
qp.a(n+1:end)  = 0;
qp.w(qp.noneg) = max(qp.w(qp.noneg),0);
qp.lb = qp.l - qp.w'*qp.w*.5;
qp.n  = n;
fprintflush(' Pruned to %d/%d with dual=%.4f \n',qp.n,length(qp.a),qp.lb);  


