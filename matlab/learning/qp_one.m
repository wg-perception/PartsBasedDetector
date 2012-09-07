% Perform one pass through current set of support vectors
function qp_one
  global qp

  MEX = true;
  %MEX = false;
  
  % Random ordering of support vectors
  I = find(qp.sv);
  I = I(randperm(length(I)));
  assert(~isempty(I));
  
  % Mex file is much faster
  if MEX,
    loss = qp_one_sparse(qp.x,qp.i,qp.b,qp.d,qp.a,qp.w,qp.noneg,qp.sv,qp.l,1,I);
  else
    sI  = sortrowsc(qp.i(:,I)',1:size(qp.i,1))';
    n   = length(I);
    idC = zeros(n,1);
    idP = zeros(n,1);
    idI = zeros(n,1);
    i0  = I(sI(1));
    C   = 1;
    num = 1;
    k   = length(qp.w);
    err = zeros(n,1);

    % Go through sorted list to compute
    % idP(i) = pointer to idC associated with I(i)
    % idC(i) = sum of alpha values with same id
    % idI(i) = pointer to some example with the same id as I(i)    
    for j = sI
      i1 = I(j);
      % Increment counter if we at new id
      if any(qp.i(:,i1) ~= qp.i(:,i0)),
        num = num + 1;
      end
      idP(j)   = num;
      idC(num) = idC(num) + qp.a(i1);
      i0 = i1;
      if qp.a(i1) > 0,
        idI(num) = i1;
      end
    end
    assert(all(idC <= C+1e-5));
    assert(all(idC >= 0-1e-5));
    
    for t = 1:n,
      i  = I(t);
      j  = idP(t);
      Ci = idC(j);
      assert(Ci <= C+1e-5);
      % Compute clamped gradient
      x1 = sparse2dense(qp.x(:,i),k);
      G  = qp.w'*x1 - double(qp.b(i));

      % Update err
      if -G > err(j),
        err(j) = -G;
      end
      
      if (qp.a(i) == 0 && G >= 0) || (Ci >= C && G <= 0),
        PG = 0;
      else
        PG = G;
      end

      % Update support vector flag
      if (qp.a(i) == 0 && G > 0),
        qp.sv(i) = 0;
      end
      
      % Check if we'd like to increase alpha but 
      % a) linear constraint is active (sum of alphas with this id == C) 
      % b) we've encountered another constraint with this id that we can decrease
      if (Ci >= C && G < -1e-12 && qp.a(i) < C && idI(j) ~= i && idI(j) > 0),
        i2 = idI(j);
        x2 = sparse2dense(qp.x(:,i2),k);
        G2 = qp.w'*x2 - double(qp.b(i2));
        numer = G - G2;
        if qp.a(i) == 0 && numer > 0,
          numer = 0;
          qp.sv(i) = 0;
        end
        if (abs(numer) > 1e-12),          
          da = -numer/(qp.d(i) + qp.d(i2) - 2*x1'*x2);
          % Clip da to box constraints
          % da > 0: a(i) = min(a(i)+da,C),  a(i2) = max(a(i2)-da,0); 
          % da < 0: a(i) = max(a(i)+da,0),  a(i2) = min(a(i2)-da,C);
          if da > 0,
            da = min(min(da,C-qp.a(i)),qp.a(i2));
          else
            da = max(max(da,-qp.a(i)),qp.a(i2)-C);
          end
          a1 = qp.a(i);
          a2 = qp.a(i2);
          qp.a(i)  = qp.a(i)  + da;
          qp.a(i2) = qp.a(i2) - da;
          assert(qp.a(i)  >= 0 && qp.a(i)  <= C);
          assert(qp.a(i2) >= 0 && qp.a(i2) <= C);
          assert(abs(a1 + a2 - (qp.a(i) + qp.a(i2))) < 1e-5);
          %obj1 = qp.l - qp.w'*qp.w*.5;
          qp.w = qp.w + da*(x1-x2);
          qp.w(qp.noneg) = max(qp.w(qp.noneg),0);
          qp.l = qp.l + da*(double(qp.b(i)) - double(qp.b(i2)));
          %obj2 = qp.l - qp.w'*qp.w*.5;
          %assert(obj2 >= obj1);
        end
      elseif (abs(PG) > 1e-12)
        % Update alpha,w, dual objective, support vector
        da = qp.a(i);
        assert(da <= Ci + 1e-5);
        maxA = max(C - (Ci-da),0);
        a1 = qp.a(i);
        qp.a(i) = min(max(qp.a(i) - G/qp.d(i),0),maxA);
        assert(qp.a(i) >= 0 && qp.a(i) <= C);
        da   = qp.a(i) - da;
        qp.w = qp.w + da*x1;
        qp.w(qp.noneg) = max(qp.w(qp.noneg),0);
        qp.l = qp.l + da*double(qp.b(i));
        idC(j) = min(max(Ci + da,0),C);
        assert(idC(j) >= 0 && idC(j) <= C);
      end
      % Record example if it can be used to satisfy a future linear constraint
      if qp.a(i) > 0,
        idI(j) = i;
      end
      %fprintf('%.5f,%.5f\n',qp.a(i),qp.w(end));
    end
    loss = sum(err);
  end

  qp_refresh();

  % Update objective
  qp.sv(qp.svfix) = 1;
  qp.lb_old = qp.lb;
  qp.lb = qp.l - qp.w'*qp.w*.5;
  qp.ub = qp.w'*qp.w*.5 + loss;
  assert(all(qp.w(qp.noneg) >= 0));
  assert(all(qp.a(1:qp.n) >= 0 - 1e-5));
  assert(all(qp.a(1:qp.n) <= 1 + 1e-5));
  %fprintf('%.16f,%d\n',qp.obj,qp.n);
return

