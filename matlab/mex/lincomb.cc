#include <math.h>
#include <stdint.h>
#include "mex.h"

// score(w,qp.x,inds)
// scores a weight vector 'w' on a set of sparse examples in qp.x at the columns specified by 'inds'
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{

  if (mxIsSingle(prhs[0]) == false) mexErrMsgTxt("Arguement 0 is not double");
  if (mxIsDouble(prhs[1]) == false) mexErrMsgTxt("Arguement 1 is not single");
  if (mxIsDouble(prhs[2]) == false) mexErrMsgTxt("Arguement 2 is not double");
  if (mxIsDouble(prhs[3]) == false) mexErrMsgTxt("Arguement 3 is not double");
  
  const float  *X = (float  *)mxGetPr(prhs[0]);
  const double *A = (double *)mxGetPr(prhs[1]);
  const double *I = (double *)mxGetPr(prhs[2]);
  const double *m = (double *)mxGetPr(prhs[3]);

  int n = mxGetNumberOfElements(prhs[2]);
  int k = mxGetM(prhs[0]);

  mxArray *mxW = mxCreateDoubleMatrix(m[0],1,mxREAL);
  double  *W   = (double *)mxGetPr(mxW);
  //printf("(%f,%d,%d)\n",m[0],n,k);
  for (int i = 0; i < n; i++) {
    const float *x = X + k*(int)(I[i]-1);
    const double a = A[(int)(I[i]-1)];
    int   xp = 1;
    // printf("(%f,%f)\n",x[0],a);
    // Iterate through blocks, and grab boundary indices using matlab's indexing
    for (int b = 0; b < x[0]; b++) { 
      int wp  = (int)x[xp++] - 1;
      int len = (int)x[xp++] - wp;
      // printf("(%f,%f,%d,%d,%d)\n",W[wp],x[xp],xp,wp,b);
      for (int j = 0; j < len; j++) {      
	W[wp++] += a * (double)x[xp++];
      }
    }
  }

  plhs[0] = mxW;
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
