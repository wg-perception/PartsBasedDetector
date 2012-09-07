#include <math.h>
#include <stdint.h>
#include "mex.h"

// score(w,qp.x,inds)
// scores a weight vector 'w' on a set of sparse examples in qp.x at the columns specified by 'inds'

#define MAX(A,B) ((A) < (B) ? (B) : (A))

static inline double score(const double *W, const float* x) {
  double y  = 0;
  int    xp = 1;
  // Iterate through blocks, and grab boundary indices using matlab's indexing
  for (int b = 0; b < x[0]; b++) { 
    int wp  = (int)x[xp++] - 1;
    int len = (int)x[xp++] - wp;
    for (int i = 0; i < len; i++) {      
      y += W[wp++] * (double)x[xp++];
    }
  }
  return y;
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{

  if (mxIsDouble(prhs[0]) == false) mexErrMsgTxt("Arguement 1 is not double");
  if (mxIsSingle(prhs[1]) == false) mexErrMsgTxt("Arguement 2 is not single");
  if (mxIsDouble(prhs[2]) == false) mexErrMsgTxt("Arguement 3 is not double");
  
  const double *W = (double  *)mxGetPr(prhs[0]);
  const float  *X = (float  *)mxGetPr(prhs[1]);
  const double *I = (double *)mxGetPr(prhs[2]);
  
  int l = mxGetNumberOfElements(prhs[2]);
  int k = mxGetM(prhs[1]);

  mxArray *mxY = mxCreateDoubleMatrix(l,1,mxREAL);
  double  *Y   = (double *)mxGetPr(mxY);
  
  for (int i = 0; i < l; i++) {    
    Y[i] = score(W,X + k*(int)(I[i]-1));
  }
  plhs[0] = mxY;
  return;
}

/*
n = 10000;
x = single(rand(n));
I = randperm(n);
I = I(1:n/5);
mask = logical(zeros(n,1));
mask(I) = 1;

tic; res = addcols(x,I); toc;

tic; res2 = double(x)*mask; toc;

tic; res3 = sum(double(x(:,I)),2); toc;
*/
