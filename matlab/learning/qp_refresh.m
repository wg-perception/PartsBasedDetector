function qp_refresh()
% Recomputes qp.w, qp.l, and qp.lb from current alpha variables
% (helpful for numerical precision issues)
global qp

MEX = true;

I = find(qp.a > 0)';
if isempty(I),
  I = 1;
end
n = length(I);

% Build 'w' by accumlating smaller numbers first 
% (better numerical stability)
[foo,ord] = sort(qp.a(I));
I = I(ord);

if MEX,
  qp.l = double(qp.b(I))'*qp.a(I);
  qp.w = lincomb(qp.x,qp.a,I,length(qp.w));
else
  qp.l = 0;
  qp.w = zeros(size(qp.w));
  k    = length(qp.w);
  for i = I,
    qp.l = qp.l + double(qp.b(i))*qp.a(i);
    k = 1;
    for j = 1:qp.x(1,i),
      i1 = qp.x(k+1,i);
      i2 = qp.x(k+2,i);
      ii = k+3:k+3+i2-i1;
      qp.w(i1:i2) = qp.w(i1:i2) + double(qp.x(ii,i))*qp.a(i);
      k  = k+3+i2-i1;
    end
  end
  l2 = double(qp.b(I))'*qp.a(I);
  w2 = lincomb(qp.x,qp.a,I,length(qp.w));
  [norm(w2 - qp.w) norm(l2 - qp.l)]
end

qp.w(qp.noneg) = max(qp.w(qp.noneg),0);
qp.lb_old = qp.lb;
qp.lb  = qp.l - qp.w'*qp.w*.5;
%fprintf(' LB=%.4f \n',qp.lb);  
if ~isempty(qp.lb_old),
  assert(qp.lb > qp.lb_old - 1e-5);
end