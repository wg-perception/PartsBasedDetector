/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    DynamicProgram.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#include <cstdio>
#include <iostream>
#include <limits>
#include "DynamicProgram.hpp"
using namespace cv;
using namespace std;

/*! @brief find nonzero elements in a matrix
 *
 * Find all nonzero elements in a matrix of type CV_8U, and return
 * the indieces (x,y) of the nonzero pixels in an array of Point
 *
 * @param binary the input image (usually the output of a comparison
 * operation such as compare(), >, ==, etc)
 * @param idx the output vector of nonzero indices
 */
void find(const Mat& binary, vector<Point>& idx) {

	assert(binary.depth() == CV_8U);
	int M = binary.rows;
	int N = binary.cols;
	for (int m = 0; m < M; ++m) {
		const unsigned char* bin_ptr = binary.ptr<unsigned char>(m);
		for (int n = 0; n < N; ++n) if (bin_ptr[n] > 0) idx.push_back(Point(n,m));
	}
}


/*! @brief Reduce a vector of matrices via indexing
 *
 * Reduce a 3D matrix (represented as a vector of matrices using cv::split() )
 * to a 2D matrix by choosing a single element from each ray cast through the
 * third dimension. This loosely emulates Matlab's matrix indexing functionality
 *
 * out[i][j] == in[i][j][ idx[i][j] ], forall i,j
 *
 * @param in the input 3D matrix
 * @param idx the index to choose at each element (idx.size() == in[k].size() forall k)
 * @param out the flatten 2D output matrix
 */
template<typename T> template<typename IT>
void DynamicProgram<T>::reducePickIndex(const vector<Mat>& in, const Mat& idx, Mat& out) {

	// error checking
	int K = in.size();
	double minv, maxv;
	minMaxLoc(idx, &minv, &maxv);
	assert(minv >= 0 && maxv < K);
	for (int k = 0; k < K; ++k) assert(in[k].size() == idx.size());

	// allocate the output array
	out.create(in[0].size(), in[0].type());

	// perform the indexing
	int M = in[0].rows;
	int N = in[0].cols;
	vector<const IT*> in_ptr(K);
	if (in[0].isContinuous()) { N = M*N; M = 1; }
	for (int m = 0; m < M; ++m) {
		IT* out_ptr = out.ptr<IT>(m);
		const int*   idx_ptr = idx.ptr<int>(m);
		for (int k = 0; k < K; ++k) in_ptr[k] = in[k].ptr<IT>(m);
		for (int n = 0; n < N; ++n) {
			out_ptr[n] = in_ptr[idx_ptr[n]][n];
		}
	}
}


/*! @brief Reduce a vector of matrices via elementwise max
 *
 * Reduce a 3D matrix (represented as a vector of matrices using cv::split() )
 * to a 2D matrix by taking the elementwise maximum across the 3D dimension.
 * Therefore in.size() == out.size() && out.channels() == 1
 *
 * @param in the input 3D matrix
 * @param maxv the output 2D matrix, containing the maximal values
 * @param maxi the output 2D matrix, containing the maximal indices
 */
template<typename T>
void DynamicProgram<T>::reduceMax(const vector<Mat>& in, Mat& maxv, Mat& maxi) {

	// TODO: flatten the input into a multi-channel matrix for faster indexing
	// error checking
	int K = in.size();
	assert(K > 1);
	for (int k = 1; k < K; ++k) assert(in[k].size() == in[k-1].size());

	// allocate the output matrices
	maxv.create(in[0].size(), in[0].type());
	maxi.create(in[0].size(), DataType<int>::type);

	int M = in[0].rows;
	int N = in[0].cols;

	vector<const T*> in_ptr(K);
	if (in[0].isContinuous()) { N = M*N; M = 1; }
	for (int m = 0; m < M; ++m) {
		T* maxv_ptr = maxv.ptr<T>(m);
		int* maxi_ptr = maxi.ptr<int>(m);
		for (int k = 0; k < K; ++k) in_ptr[k] = in[k].ptr<T>(m);
		for (int n = 0; n < N; ++n) {
			T v = -numeric_limits<T>::infinity();
			int i = 0;
			for (int k = 0; k < K; ++k) if (in_ptr[k][n] > v) { i = k; v = in_ptr[k][n]; }
			maxi_ptr[n] = i;
			maxv_ptr[n] = v;
		}
	}
}


/*! @brief the square of an integer
 *
 * @param x the input
 * @return x*x
 */
static inline int square(int x) { return x*x; }


/*! @brief Generalized 1D distance transform
 *
 * This method performs the 1D distance transform across the rows of a matrix.
 * It is called twice internally by distanceTransform(), once across the rows
 * and once down the columns (transposed to rows)
 *
 * Only quadratic distance functions are handled
 * y = a.x^2 + b.x
 *
 * @param src pointer to the start of the source data
 * @param dst pointer to the start of the destination data
 * @param ptr pointer to indices
 * @param n the total number of rows
 * @param a the quadratic coefficient
 * @param b the linear coefficient
 */
template<typename T>
void DynamicProgram<T>::distanceTransform1D(const T* src, T* dst, int* ptr, int n, T a, T b) {

	int * const v = new int[n];
	T   * const z = new T[n+1];
	int k = 0;
	v[0] = 0;
	z[0] = -numeric_limits<T>::infinity();
	z[1] = +numeric_limits<T>::infinity();
	for (int q = 1; q <= n-1; ++q) {
	    T s = ((src[q] - src[v[k]]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
	    while (s <= z[k] && k > 0) {
			// Update pointer
			k--;
			s  = ((src[q] - src[v[k]]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
	    }
	    k++;
	    v[k]   = q;
	    z[k]   = s;
	    z[k+1] = +numeric_limits<T>::infinity();
	}

	k = 0;
	for (int q = 0; q <= n-1; ++q) {
		while (z[k+1] < q) k++;
		dst[q] = a*square(q-v[k]) + b*(q-v[k]) + src[v[k]];
		ptr[q] = v[k];
	}

	delete [] v;
	delete [] z;
}


/*! @brief Generalized distance transform
 *
 * 1-Dimensional generalized distance transform based on the paper:
 * P. Felzenszwalb and D. Huttenlocher, "Distance Transforms of Sampled Functions,"
 * Cornell Technical Report, 2004
 *
 * This is used to reduce the complexity of the dynamic program, namely when all
 * of the cost functions are quadratic. The 2D distance transform is broken down
 * into two 1D transforms since the operation is separable
 *
 * @param score_in the input score
 * @param w the quadratic weights
 * @param score_out the distance transformed score
 * @param Ix the distances in the x direction
 * @param Iy the distances in the y direction
 */
template<typename T>
void DynamicProgram<T>::distanceTransform(const Mat& score_in, const vector<float> w, Mat& score_out, Mat& Ix, Mat& Iy) {

	// get the dimensionality of the score
	int M = score_in.rows;
	int N = score_in.cols;

	// get the learned quadratic coefficients
	float ax = w[0];
	float bx = w[1];
	float ay = w[2];
	float by = w[3];

	// allocate the output and working matrices
	score_out.create(Size(M, N), DataType<T>::type);
	Ix.create(Size(N, M), DataType<int>::type);
	Iy.create(Size(M, N), DataType<int>::type);

	Mat score_tmp(Size(N, M), DataType<T>::type);
	Mat row(Size(N, 1), DataType<int>::type);

	// compute the distance transform across the rows
	for (int m = 0; m < M; ++m) {
		distanceTransform1D(score_in.ptr<T>(m), score_tmp.ptr<T>(m), Ix.ptr<int>(m), N, -ax, -bx);
	}

	// transpose the intermediate matrices
	transpose(score_tmp, score_tmp);

	// compute the distance transform down the columns
	for (int n = 0; n < N; ++n) {
		distanceTransform1D(score_tmp.ptr<T>(n), score_out.ptr<T>(n), Iy.ptr<int>(n), M, -ay, -by);
	}

	// transpose back to the original layout
	transpose(score_out, score_out);
	transpose(Iy, Iy);

	// get argmins
	// FIXME: this miiiight be wrong! Check this against the original code if there are bugs in the dynamic program
	int * const row_ptr = row.ptr<int>(0);
	for (int m = 0; m < M; ++m) {
		int* Iy_ptr = Iy.ptr<int>(m);
		int* Ix_ptr = Ix.ptr<int>(m);
		for (int n = 0; n < N; ++n) {
			row_ptr[n] = Iy_ptr[Ix_ptr[n]];
		}
		for (int n = 0; n < N; ++n) {
			Iy_ptr[n] = row_ptr[n];
		}
	}
}

template<typename T>
void DynamicProgram<T>::minRecursive(Part& self, Part& parent, vector<Mat>& scores, int nparts, int scale) {

	// if this is not a leaf node, request the child messages
	if (!self.isLeaf()) {
		for (int c = 0; c < self.children().size(); ++c) minRecursive(self.children()[c], self, scores, nparts, scale);
	}

	// intermediate results
	vector<Mat> scoresi;
	vector<Mat> Ixi;
	vector<Mat> Iyi;
	// final results
	vector<Mat> Ix;
	vector<Mat> Iy;
	vector<Mat> Ik;

	// calculate the score of each of the mixtures
	int nmixtures = self.nmixtures();
	int Ns = nparts*nmixtures*scale + nmixtures*self.pos();
	int Np = nparts*nmixtures*scale + nmixtures*parent.pos();
	for (int m = 0; m < nmixtures; ++m) {
		// raw score outputs
		Mat score_dt, Ix_dt, Iy_dt;
		Mat score_in = scores[Ns+m];
		// compute the distance transform
		distanceTransform(score_in, self.w()[m], score_dt, Ix_dt, Iy_dt);

		// get the anchor position
		Point anchor = self.anchor();

		// calculate a valid region of interest for the scores
		int X = score_in.cols;
		int Y = score_in.rows;
		int xmin = std::max(anchor.x,   0);
		int ymin = std::max(anchor.y,   0);
		int xmax = std::min(anchor.x+X, X);
		int ymax = std::min(anchor.y+Y, X);
		int xoff = std::max(-anchor.x,  0);
		int yoff = std::max(-anchor.y,  0);

		// shift the score by the Part's offset from its parent
		Mat score = -numeric_limits<T>::infinity() * Mat::ones(score_dt.size(), score_dt.type());
		Mat Ixm   = Mat::zeros(Ix_dt.size(), Ix_dt.type());
		Mat Iym   = Mat::zeros(Iy_dt.size(), Iy_dt.type());
		score(Range(xoff, xoff+xmax-xmin), Range(yoff, yoff+ymax-ymin)) = score_dt(Range(xmin, xmax), Range(ymin, ymax));
		Ixm(Range(xoff, xoff+xmax-xmin), Range(yoff, yoff+ymax-ymin)) = Ix_dt(Range(xmin, xmax), Range(ymin, ymax));
		Iym(Range(xoff, xoff+xmax-xmin), Range(yoff, yoff+ymax-ymin)) = Iy_dt(Range(xmin, xmax), Range(ymin, ymax));

		// push the scores onto the intermediate vectors
		scoresi.push_back(score);
		Ixi.push_back(Ixm);
		Iyi.push_back(Iym);
	}

	// at each parent location, for each parent mixture, compute the best child mixture
	for (int m = 0; m < nmixtures; ++m) {
		vector<float> bias = self.bias()[m];
		vector<Mat> weighted;
		Mat I;
		// weight each of the child scores
		for (int mm = 0; mm < nmixtures; ++mm) {
			weighted.push_back(scores[mm] + bias[mm]);
		}

		// compute the max over the mixtures
		Mat maxv, maxi;
		reduceMax(weighted, maxv, maxi);

		// choose the best indices
		Mat Ixm, Iym;
		reducePickIndex<int>(Ixi, maxi, Ixm);
		reducePickIndex<int>(Iyi, maxi, Iym);
		Ix.push_back(Ixm);
		Iy.push_back(Iym);
		Ik.push_back(maxi);
		scores[Np+m] = maxv;
	}
}




/*! @brief Get the min of a dynamic program
 *
 * Get the min of a dynamic program by starting at the leaf nodes,
 * and passing the scores up to the root node. This is essentially
 * a tail recursive algorithm which we can unfold since we know that
 * the parts are sorted from the root to the leaves
 *
 * The algorithm involves 3 steps:
 * 		(1) Apply distance transform
 * 		(2) Shift by the anchor position of the part wrt the parent
 * 		(3) Downsample if necessary
 *
 * @param parts the parts tree, referenced by the root
 * @param scores the probability densities (pdfs) of part locations (fine to coarse)
 * @param Ix the detection indices in the x direction
 * @param Iy the detection indices in the y direction
 * @param Ik the best mixture at each pixel
 * @param rootv the root scores, across scale
 * @param rooti the root indices, across scale
 *
 */
template<typename T>
void DynamicProgram<T>::min(Parts& parts, vector2DMat& scores, vector4DMat& Ix, vector4DMat& Iy, vector4DMat& Ik, vector2DMat& rootv, vector2DMat& rooti) {

	// initialize the outputs, preallocate vectors to make them thread safe
	// TODO: better initialisation of Ix, Iy, Ik
	int nscales = scores.size();
	Ix.resize(nscales);
	Iy.resize(nscales);
	Ik.resize(nscales);
	rootv.resize(nscales, vectorMat(parts.ncomponents()));
	rooti.resize(nscales, vectorMat(parts.ncomponents()));

	// for each scale, and each component, update the scores through message passing
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < nscales; ++n) {
		Ix[n].resize(parts.ncomponents());
		Iy[n].resize(parts.ncomponents());
		Ik[n].resize(parts.ncomponents());

		for (int c = 0; c < parts.ncomponents(); ++c) {
			Ix[n][c].resize(parts.nparts(c));
			Iy[n][c].resize(parts.nparts(c));
			Ik[n][c].resize(parts.nparts(c));

			for (int p = parts.nparts(c)-1; p > 0; --p) {

				// get the component part (which may have multiple mixtures associated with it)
				ComponentPart cpart = parts.component(c, p);
				int nmixtures       = cpart.nmixtures();
				Ix[n][c][p].resize(nmixtures);
				Iy[n][c][p].resize(nmixtures);
				Ik[n][c][p].resize(nmixtures);

				// intermediate results for mixtures of this part
				vector<Mat> scoresp;
				vector<Mat> Ixp;
				vector<Mat> Iyp;

				for (int m = 0; m < nmixtures; ++m) {
					// raw score outputs
					Mat score_dt, Ix_dt, Iy_dt;
					Mat score_in = cpart.score(scores[n], m);

					// compute the distance transform
					distanceTransform(score_in, cpart.defw(m), score_dt, Ix_dt, Iy_dt);

					// get the anchor position
					Point anchor = cpart.anchor(m);

					// calculate a valid region of interest for the scores
					int X = score_in.cols;
					int Y = score_in.rows;
					int xmin = std::max(std::min(anchor.x, X), 0);
					int ymin = std::max(std::min(anchor.y, Y), 0);
					int xmax = std::min(std::max(anchor.x+X, 0), X);
					int ymax = std::min(std::max(anchor.y+Y, 0), Y);
					int xoff = std::max(-anchor.x,    0);
					int yoff = std::max(-anchor.y,    0);

					// shift the score by the Part's offset from its parent
					Mat scorem = -numeric_limits<T>::infinity() * Mat::ones(score_dt.size(), score_dt.type());
					Mat Ixm    = Mat::zeros(Ix_dt.size(), Ix_dt.type());
					Mat Iym    = Mat::zeros(Iy_dt.size(), Iy_dt.type());
					Mat score_dt_range 	= score_dt(Range(ymin, ymax),         Range(xmin, xmax));
					Mat score_range    	= scorem(Range(yoff, yoff+ymax-ymin), Range(xoff, xoff+xmax-xmin));
					Mat Ix_dt_range 	= Ix_dt(Range(ymin, ymax),            Range(xmin, xmax));
					Mat Ixm_range 		= Ixm(Range(yoff, yoff+ymax-ymin),    Range(xoff, xoff+xmax-xmin));
					Mat Iy_dt_range 	= Iy_dt(Range(ymin, ymax),            Range(xmin, xmax));
					Mat Iym_range 		= Iym(Range(yoff, yoff+ymax-ymin),    Range(xoff, xoff+xmax-xmin));
					score_dt_range.copyTo(score_range);
					Ix_dt_range.copyTo(Ixm_range);
					Iy_dt_range.copyTo(Iym_range);

					// push the scores onto the intermediate vectors
					scoresp.push_back(scorem);
					Ixp.push_back(Ixm);
					Iyp.push_back(Iym);
				}

				nmixtures = cpart.parent().nmixtures();
				for (int m = 0; m < nmixtures; ++m) {
					vector<Mat> weighted;
					// weight each of the child scores
					// TODO: More elegant way of handling bias
					for (int mm = 0; mm < cpart.nmixtures(); ++mm) {
						weighted.push_back(scoresp[mm] + cpart.bias(mm)[m]);
					}
					// compute the max over the mixtures
					Mat maxv, maxi;
					reduceMax(weighted, maxv, maxi);


					// choose the best indices
					Mat Ixm, Iym;
					reducePickIndex<int>(Ixp, maxi, Ixm);
					reducePickIndex<int>(Iyp, maxi, Iym);
					Ix[n][c][p][m] = Ixm;
					Iy[n][c][p][m] = Iym;
					Ik[n][c][p][m] = maxi;



					// update the parent's score
					cpart.parent().score(scores[n],m) += maxv;
				}
			}
			// add bias to the root score and find the best mixture
			ComponentPart root = parts.component(c);
			T bias = root.bias(0)[0];
			vector<Mat> weighted;
			// weight each of the child scores
			for (int m = 0; m < root.nmixtures(); ++m) {
				weighted.push_back(root.score(scores[n],m) + bias);
			}
			reduceMax(weighted, rootv[n][c], rooti[n][c]);
		}
	}
}


/*! @brief get the argmin of a dynamic program
 *
 * Get the minimum argument of a dynamic program by traversing down the tree of
 * a dynamic program, returning the locations of the best nodes
 * @param parts the tree of parts, referenced by the root
 * @param rootv the root scores, across scale
 * @param rooti the root indices, across scale
 * @param scales the scales (used to calculate bounding box size)
 * @param Ix the detection indices in the x direction
 * @param Iy the detection indices in the y direction
 * @param Ik the best mixture at each pixel
 * @param candidates
 */
template<typename T>
void DynamicProgram<T>::argmin(Parts& parts, const vector2DMat& rootv, const vector2DMat& rooti, const vectorf scales, const vector4DMat& Ix, const vector4DMat& Iy, const vector4DMat& Ik, vector<Candidate>& candidates) {

	// for each scale, and each component, traverse back down the tree to retrieve the part positions
	int nscales = scales.size();
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < nscales; ++n) {
		T scale = scales[n];
		for (int c = 0; c < parts.ncomponents(); ++c) {

			// get the scores and indices for this tree of parts
			const vector2DMat& Iknc = Ik[n][c];
			const vector2DMat& Ixnc = Ix[n][c];
			const vector2DMat& Iync = Iy[n][c];
			int nparts = parts.nparts(c);

			// threshold the root score
			Mat over_thresh = rootv[n][c] > thresh_;
			Mat rootmix     = rooti[n][c];
			vectorPoint inds;
			find(over_thresh, inds);

			for (int i = 0; i < inds.size(); ++i) {
				Candidate candidate;
				vectori     xv(nparts);
				vectori     yv(nparts);
				vectori     mv(nparts);
				for (int p = 0; p < nparts; ++p) {
					ComponentPart part = parts.component(c, p);
					// calculate the child's points from the parent's points
					int x, y, m;
					if (part.isRoot()) {
						x = xv[0] = inds[i].x;
						y = yv[0] = inds[i].y;
						m = mv[0] = rootmix.at<int>(inds[i]);
					} else {
						int idx = part.parent().self();
						x = xv[idx];
						y = yv[idx];
						m = mv[idx];
						xv[p] = Ixnc[p][m].at<int>(y,x);
						yv[p] = Iync[p][m].at<int>(y,x);
						mv[p] = Iknc[p][m].at<int>(y,x);
					}

					// calculate the bounding rectangle and add it to the Candidate
					Point ptwo = Point(2,2);
					Point pone = Point(1,1);
					Point xy1 = (Point(xv[p],yv[p])-ptwo)*scale;
					Point xy2 = xy1 + Point(part.xsize(m), part.ysize(m))*scale - pone;
					if (part.isRoot()) candidate.addPart(Rect(xy1, xy2), rootv[n][c].at<T>(inds[i]));
					else candidate.addPart(Rect(xy1, xy2), 0.0);
				}
				#ifdef _OPENMP
				#pragma omp critical(addcandidate)
				#endif
				{
					candidates.push_back(candidate);
				}
			}
		}
	}
}

// declare all specializations of the template (this must be the last declaration in the file)
template class DynamicProgram<float>;
template class DynamicProgram<double>;

