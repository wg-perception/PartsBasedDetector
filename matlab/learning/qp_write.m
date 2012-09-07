% qp_write(ex)
% where ex.id   = K X 1
%       ex.blocks(j).i = starting index of j^th feature block
%       ex.blocks(j).x = feature block
%       final feature  = [ex.blocks(:).x]
%
% Write ex to a QP of the form:
% min_{w,e}  ||(w-w0)*r||^2 + sum_i c_i e_i
%      s.t.  w x_ij >= 1 - e_i 
%
% We can write the above QP in "standard" form
% with the following substitution: v = (w-w0)*r
% min_{v,e}  ||v||^2 + sum_i e_i
% s.t. v x'_ij >= b'_ij - e_i   
%
% where  x'_ij = c_i*(x_ij/r)
%        b'_ij = c_i*(1 - w0*x_ij)
function qp_write(ex)
  global qp;
  
  if qp.n == length(qp.a),
    return;
  end
  
  label = ex.id(1) > 0;

  if label,
    C = qp.Cpos;
  else
    C = qp.Cneg;
  end
 
  % Ensure there are no duplicate blocks
  is = sort([ex.blocks.i]);
  assert(~any(is(2:end) == is(1:end-1)));
   
  % Sparsely compute these 3 quantities 
  % x    = C*(label*feat ./ qp.wreg)  
  % bias = C*(1 - qp.w0'*label*feat)
  % norm = x'*x
  bias = 1;
  norm = 0;
  qp.n = qp.n + 1;
  i    = qp.n;
  j    = 1;
  qp.x(:,i) = 0;
  qp.x(j,i) = length(ex.blocks);
  
  for b = ex.blocks,
    n  = numel(b.x);
    i1 = b.i;
    i2 = i1 + n - 1;
    is = i1:i2;
    x  = reshape(b.x,n,1);
    if ~label,
      x = -x;
    end

    bias = bias - qp.w0(is)'*x;
    x    = C * x ./ qp.wreg(is);

    qp.x(j+1,i) = i1;
    qp.x(j+2,i) = i2;
    qp.x(j+3:j+3+i2-i1,i) =  x;

    norm = norm + x'*x;
    
    j = j+3+i2-i1;
  end
  
  qp.d(i)   = norm;
  qp.b(i)   = C*bias;
  qp.i(:,i) = ex.id;
  qp.sv(i)  = 1;
 
