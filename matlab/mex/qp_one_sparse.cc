#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"

#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define INDEX(X,I) (*(int *)(X + I*mm + m))
int n, m, mm;

// Comparison function for sorting function in sumAlpha function
int comp(const void *a, const void *b) {
  return memcmp((int32_t *)a,(int32_t *)b,m*sizeof(int32_t));
}

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

static inline double dot(const float *x, const float *y) {
  double res = 0;
  int xnum = (int)x[0];
  int ynum = (int)y[0];

  //  b: block number
  //  i: position in sparse vector
  //  j: position in dense vector
  // j1: start position in dense vector (matlab index)
  // j2: end   position in dense vector (matlab index)
  int yb=0, xb=0;
  int yi=1, xi=1;
  int yj1 = (int)y[yi++];
  int yj2 = (int)y[yi++];
  int xj1 = (int)x[xi++];
  int xj2 = (int)x[xi++];

  while(1) {
    // Find intersecting indices
    if (xj2 >= yj1 && yj2 >= xj1) {
      int j1 = MAX(xj1,yj1);
      int j2 = MIN(xj2,yj2);
      int xp = xi + j1 - xj1;
      int yp = yi + j1 - yj1;
      for (int k=0;k < j2-j1+1;k++) {
	res += (double)x[xp++] * (double)y[yp++];
      }
    }
    // Increment x or y pointer
    if (yj2 <= xj2) {
      if (++yb >= ynum) break;
      yi += yj2-yj1+1;
      yj1 = y[yi++];
      yj2 = y[yi++];
    } else {
      if (++xb >= xnum) break;
      xi += xj2-xj1+1;
      xj1 = x[xi++];
      xj2 = x[xi++];
    }
  }   
  return res;
}

static inline void add(double *W, const float* x, const double a) {
  int xp = 1;
  // Iterate through blocks, and grab boundary indices using matlab's indexing
  for (int b = 0; b < x[0]; b++) { 
    int wp  = (int)x[xp++] - 1;
    int len = (int)x[xp++] - wp;
    for (int i = 0; i < len; i++) {
      W[wp++] += a * (double)x[xp++];
    }
    //printf("(%d,%d)",wp,len);
  }
}

// idC(idP)[i] is the sum of alpha value for examples with ID(:,I(i))
// idI[i] is a pointer to some example with the same id as example I[i]
void sumAlpha(const int32_t *ID, const double* A, const double *I,double *idC, int *idP, int *idI) {
  mm = m + sizeof(int)/sizeof(int32_t);
  int32_t *sID = (int32_t *)mxCalloc(mm*n,sizeof(int32_t));

 // Create a matrix of selected ids (given with matlab indexing)
 // that will be sorted, appending the position at end
  for (int j = 0; j < n; j++) {
    int j0 = (I[j]-1)*m;
    int j1 =  j*mm;
    for (int i = 0; i < m; i++) {
      sID[j1+i] = ID[j0+i];
    }
    INDEX(sID,j) = j;
  }

  // Sort
  qsort(sID,n,mm*sizeof(int32_t),comp);

  // Go through sorted list, adding alpha values of examples with identical ids
  int i0  = I[INDEX(sID,0)]-1;
  int num = 0;
  for (int t = 0; t < n; t++) {
    int j  = INDEX(sID,t);
    int i1 = I[j] - 1;
    if (memcmp( ID + m*i1, ID + m*i0,m*sizeof(int32_t)) != 0) 
      num++;   
    idP[j]    = num + 1;
    idC[num] += A[i1];    
    i0       = i1;
    if (A[i1] > 0) {
      idI[num] = i1 + 1;
    }
  }
  mxFree(sID);
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{

  const float   *X  = (float   *)mxGetPr(prhs[0]);
  const int32_t *ID = (int32_t *)mxGetPr(prhs[1]);
  const float   *B  = (float   *)mxGetPr(prhs[2]);
  const double  *D  = (double  *)mxGetPr(prhs[3]);
  double        *A  = (double  *)mxGetPr(prhs[4]);
  double        *W  = (double *)mxGetPr(prhs[5]);
  const uint32_t *noneg = (uint32_t *)mxGetPr(prhs[6]);
  bool   *SV = (bool   *)mxGetPr(prhs[7]);
  double *L  = (double *)mxGetPr(prhs[8]);
  double  C  = (double  )mxGetScalar(prhs[9]);
  double *I  = (double *)mxGetPr(prhs[10]);

  if (nrhs < 10) mexErrMsgTxt("Incorrect number of input arguments.");
  if (mxIsSingle(prhs[0])  == false) mexErrMsgTxt("Argument 0 is not single.");
  if ( mxIsInt32(prhs[1])  == false) mexErrMsgTxt("Argument 1 is not int32.");
  if (mxIsSingle(prhs[2])  == false) mexErrMsgTxt("Argument 2 is not single.");
  if (mxIsDouble(prhs[3])  == false) mexErrMsgTxt("Argument 3 is not double.");
  if (mxIsDouble(prhs[4])  == false) mexErrMsgTxt("Argument 4 is not double.");
  if (mxIsDouble(prhs[5])  == false) mexErrMsgTxt("Argument 5 is not double.");
  if (mxIsUint32(prhs[6])  == false) mexErrMsgTxt("Argument 6 is not uint32.");
  if (mxIsLogical(prhs[7]) == false) mexErrMsgTxt("Argument 7 is not logical.");
  if (mxIsDouble(prhs[8])  == false) mexErrMsgTxt("Argument 8 is not double.");
  if (mxIsDouble(prhs[9])  == false) mexErrMsgTxt("Argument 9 is not double.");
  if (mxIsDouble(prhs[10]) == false) mexErrMsgTxt("Argument 10 is not double.");
  
  int k = mxGetM(prhs[0]);
  int p = MAX(mxGetN(prhs[6]),mxGetM(prhs[6]));  
  n = MAX(mxGetN(prhs[10]),mxGetM(prhs[10]));  
  m = mxGetM(prhs[1]);
  

  // idC(idP(i)) is the sum of alpha value for examples with ID(:,I(i))
  // err(idP(i)) is the maximum loss for examples with ID(:,I(i))
  double *err = (double *)mxCalloc(n,sizeof(double));
  double *idC = (double *)mxCalloc(n,sizeof(double)); 
  int    *idP = (int    *)mxCalloc(n,sizeof(int));
  int    *idI = (int    *)mxCalloc(n,sizeof(int));

  sumAlpha(ID,A,I,idC,idP,idI);

  //printf("Intro: (m,n,C) = (%d,%d,%g)\n",m,n,C);
  for (int cnt = 0; cnt < n; cnt++) {
    // Use C indexing
    int i = (int)  I[cnt] - 1;
    int j = (int)idP[cnt] - 1;
    const float *x = X + k*i;
    // The following two lines are useful for violations of
    // 0<=Ai<=C and Ai<=Ci<=C due to precision issues
    A[i]      = MAX(MIN(A[i],  C),   0);
    double Ci = MAX(MIN(idC[j],C),A[i]);
    double G  = score(W,x) - (double)B[i];
    double PG = G;
    
    if ((A[i] == 0 && G >= 0) || (Ci >= C && G <= 0)) {
      PG = 0;
    }
    
    // Update error
    if (-G > err[j]) {
      err[j] = -G; 
    }

    // Update support vector flag
    if (A[i] == 0 && G > 0) {
      SV[i] = false;
    }
 
    //printf("[%d,%d,%g,%g,%g]\n",cnt,i,G,PG,A[i]);
    if (Ci >= C && G < -1e-12 && A[i] < C && idI[j]-1 != i && idI[j] > 0) {
      int i2 = idI[j]-1;
      const float *x2 = X + k*i2;

      // G = G - G2, where G2 = w*x2 - b2
      G -= (score(W,x2) - (double)B[i2]);

      if (A[i] == 0 && G > 0) {
	G = 0;
	SV[i] = false;
      }
      
      if (G > 1e-12 || G < -1e-12) {

	double dA = -G / (D[i] + D[i2] - 2*dot(x,x2));
	
	//printf("[%d,%g,%d,%g,%g,%g]\n",i,A[i],i2,A[i2],G,dA);
	
	if (dA > 0) {
	  dA = MIN(MIN(dA,C - A[i]),A[i2]);
	} else {
	  dA = MAX(MAX(dA,-A[i]),A[i2]-C);
	} 
	A[i]  = A[i]  + dA;
	A[i2] = A[i2] - dA;
	L[0] += dA * ((double)B[i] - (double)B[i2]);
	// w = w + da*(x-x2)
	add(W, x, dA);
	add(W,x2,-dA);
	for (int d = 0; d < p; d++) {
	  W[noneg[d]-1] = MAX( W[noneg[d]-1], 0);
	}
	//printf("[%g,%g,%g,%g,%g,%g]\n",dA,A[i],A[i2],B[i],B[i2],L[0]);
      }
    }
    else if (PG > 1e-12 || PG < -1e-12) {
      double dA   = A[i];
      double maxA = C - (Ci - dA);
      A[i]  = MIN ( MAX ( A[i] - G/D[i], 0 ) , maxA);
      dA    = A[i] - dA;
      L[0] += dA * (double) B[i];
      idC[j] = MIN ( MAX ( Ci + dA, 0 ), C);
      //printf("%g,%g,%g,%g\n",A[i],B[i],dA,*L);
      add(W,x,dA);
      // Ensure nonegativity of certain weights given by MATLAB indexing
      for (int d = 0; d < p; d++) {
	//printf("%d,%d,%g\n",d,noneg[d]-1,W[noneg[d]-1]);
	W[noneg[d]-1] = MAX( W[noneg[d]-1], 0);
      }
    }    
    //Record example if it can be used to satisfy a future linear constraint
    //(use matlab indexing)
    if (A[i] > 0) {
      idI[j] = i + 1;
    }
  }

  // Compute total error
  double sum = 0;  
  for (int i = 0; i < n; i++) {
    sum += err[i];
  }
  plhs[0] = mxCreateDoubleScalar(sum);

  mxFree(err);
  mxFree(idC);
  mxFree(idP);
  mxFree(idI);
}

