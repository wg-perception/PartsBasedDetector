function qp_opt(tol,iter)
% qp_opt(tol,iter)
% Optimize QP until relative difference between lower and upper bound is below 'tol'

global qp;

if nargin < 1,
  tol = .05;
end

if nargin < 2,
  iter = 1000;
end

% Recompute qp.w in case of numerical precision issues
qp_refresh();

C = 1;
I = 1:qp.n;
[id,J] = sortrows(qp.i(:,I)');
id     = id';
eqid   = [0 all(id(:,2:end) == id(:,1:end-1),1)];

slack = qp.b(I) - score(qp.w,qp.x,I);
loss  = computeloss(slack(J),eqid);
ub    = qp.w'*qp.w*.5 + C*loss; 
lb    = qp.lb;
qp.sv(I) = 1;
fprintflush('\n LB=%.4f,UB=%.4f [',lb,ub);
% Iteratively apply coordinate descent, pruning active set (support vectors)
% If we've possible converged over active set
% 1) Compute true upper bound over full set
% 2) If we haven't actually converged, 
%    reinitialize optimization to full set
for t = 1:iter,
  qp_one;
  lb     = qp.lb;
  ub_est = min(qp.ub,ub);
  fprintflush('.');
  if lb > 0 && 1 - lb/ub_est < tol,
    slack = qp.b(I) - score(qp.w,qp.x,I);
    loss  = computeloss(slack(J),eqid);
    ub    = min(ub,qp.w'*qp.w*.5 + C*loss);  
    if 1 - lb/ub < tol,
      break;
    end
    qp.sv(I) = 1;
  end
  %fprintf('t=%d: LB=%.4f,UB_true=%.5f,UB_est=%.5f,#SV=%d\n',t,lb,ub,ub_est,sum(qp.sv));
end

qp.ub = ub;
fprintflush('] LB=%.4f,UB=%.4f\n',lb,ub);


function loss = computeloss(slack,eqid)
% Zero-out scores that aren't the greatest violated constraint for an id
% eqid(i) = 1 if x(i) and x(i-1) are from the same id
% eqid(1) = 0
% v is the best value in the current block
% i is a ptr to v
% j is a ptr to the example we are considering

err = logical(zeros(size(eqid)));
for j = 1:length(err),
  % Are we at a new id?
  % If so, update i,v
  if eqid(j) == 0,
    i = j;
    v = slack(i);
    if v > 0,
      err(i) = 1;
    else
      v = 0;
    end
    % Are we at a new best in this set of ids?
    % If so, update i,v and zero out previous best
  elseif slack(j) > v 
    err(i) = 0;
    i = j;
    v = slack(i);
    err(i) = 1;
  end
end

loss = sum(slack(err));  
