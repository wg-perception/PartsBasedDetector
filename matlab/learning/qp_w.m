function w = qp_w
  global qp
  w = qp.w ./ qp.wreg + qp.w0;
  
