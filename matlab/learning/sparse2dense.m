function y = sparse2dense(x,n)
% Turn a sparse vector into a block sparse vector

y = zeros(n,1);
j = 1;
for i = 1:x(1),
  i1 = x(j+1);
  i2 = x(j+2);
  y(i1:i2) = double(x(j+3:j+3+i2-i1));
  j  = j+3+i2-i1;
end
