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
#include "Math.hpp"
#include "DynamicProgram.hpp"
using namespace cv;
using namespace std;

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
 * @param N the total number of rows
 * @param a the quadratic coefficient
 * @param b the linear coefficient
 * @param os the anchor offset
 */
template<typename T>
inline void DynamicProgram<T>::distanceTransform1D(const T* src, T* dst, int* ptr, unsigned int N, T a, T b, int os) {

	int * const v = new int[N];
	T   * const z = new T[N+1];
	int k = 0;
	v[0] = 0;
	z[0] = -numeric_limits<T>::infinity();
	z[1] = +numeric_limits<T>::infinity();
	for (unsigned int q = 1; q < N; ++q) {
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
	for (unsigned int q = 0; q < N; ++q) {
		while(z[k+1] < q) k++;
		dst[q] = a*square(q-v[k]) + b*(q-v[k]) + src[v[k]];
		ptr[q] = v[k];
	}

	k = 0;
	for (unsigned int q = 0; q < N; ++q) {
		while (z[k+1] < os) k++;
		dst[q] = a*square(os-v[k]) + b*(os-v[k]) + src[v[k]];
		ptr[q] = v[k];
		os++;
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
 * @param os the anchor offset of the child from the parent
 * @param score_out the distance transformed score
 * @param Ix the distances in the x direction
 * @param Iy the distances in the y direction
 */
template<typename T>
void DynamicProgram<T>::distanceTransform(const Mat& score_in, const vectorf w, Point os, Mat& score_out, Mat& Ix, Mat& Iy) {

	// get the dimensionality of the score
	const unsigned int M = score_in.rows;
	const unsigned int N = score_in.cols;

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
	for (unsigned int m = 0; m < M; ++m) {
		distanceTransform1D(score_in.ptr<T>(m), score_tmp.ptr<T>(m), Ix.ptr<int>(m), N, -ax, -bx, os.x);
	}

	// transpose the intermediate matrices
	transpose(score_tmp, score_tmp);

	// compute the distance transform down the columns
	for (unsigned int n = 0; n < N; ++n) {
		distanceTransform1D(score_tmp.ptr<T>(n), score_out.ptr<T>(n), Iy.ptr<int>(n), M, -ay, -by, os.y);
	}

	// transpose back to the original layout
	transpose(score_out, score_out);
	transpose(Iy, Iy);

	// get argmins
	int * const row_ptr = row.ptr<int>(0);
	for (unsigned int m = 0; m < M; ++m) {
		int* Iy_ptr = Iy.ptr<int>(m);
		int* Ix_ptr = Ix.ptr<int>(m);
		for (unsigned int n = 0; n < N; ++n) {
			row_ptr[n] = Iy_ptr[Ix_ptr[n]];
		}
		for (unsigned int n = 0; n < N; ++n) {
			Iy_ptr[n] = row_ptr[n];
		}
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
	const unsigned int nscales = scores.size();
	const unsigned int ncomponents = parts.ncomponents();
	Ix.resize(nscales, vector3DMat(ncomponents));
	Iy.resize(nscales, vector3DMat(ncomponents));
	Ik.resize(nscales, vector3DMat(ncomponents));
	rootv.resize(nscales, vectorMat(ncomponents));
	rooti.resize(nscales, vectorMat(ncomponents));

	// for each scale, and each component, update the scores through message passing
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (unsigned int nc = 0; nc < nscales*ncomponents; ++nc) {

		// calculate the inner loop variables from the dual variables
		const unsigned int n = floor(nc / ncomponents);
		const unsigned int c = nc % ncomponents;

		// allocate the inner loop variables
		Ix[n][c].resize(parts.nparts(c));
		Iy[n][c].resize(parts.nparts(c));
		Ik[n][c].resize(parts.nparts(c));
		vectorMat ncscores(scores[n].size());

		for (int p = parts.nparts(c)-1; p > 0; --p) {

			// get the component part (which may have multiple mixtures associated with it)
			ComponentPart cpart = parts.component(c, p);
			unsigned int nmixtures       = cpart.nmixtures();
			Ix[n][c][p].resize(nmixtures);
			Iy[n][c][p].resize(nmixtures);
			Ik[n][c][p].resize(nmixtures);

			// intermediate results for mixtures of this part
			vectorMat scoresp;
			vectorMat Ixp;
			vectorMat Iyp;

			for (unsigned int m = 0; m < nmixtures; ++m) {

				// raw score outputs
				Mat score_in, score_dt, Ix_dt, Iy_dt;
				if (cpart.score(ncscores, m).empty()) {
					score_in = cpart.score(scores[n], m);
				} else {
					score_in = cpart.score(ncscores, m);
				}

				// get the anchor position
				Point anchor = cpart.anchor(m);

				// compute the distance transform
				distanceTransform(score_in, cpart.defw(m), anchor, score_dt, Ix_dt, Iy_dt);
				//score_in.copyTo(score_dt);
				//Ix_dt = Mat::zeros(score_dt.size(), DataType<int>::type);
				//Iy_dt = Mat::zeros(score_dt.size(), DataType<int>::type);
				scoresp.push_back(score_dt);
				Ixp.push_back(Ix_dt);
				Iyp.push_back(Iy_dt);
			}

			nmixtures = cpart.parent().nmixtures();
			for (unsigned int m = 0; m < nmixtures; ++m) {
				vectorMat weighted;
				// weight each of the child scores
				// TODO: More elegant way of handling bias
				for (unsigned int mm = 0; mm < cpart.nmixtures(); ++mm) {
					weighted.push_back(scoresp[mm] + cpart.bias(mm)[m]);
				}
				// compute the max over the mixtures
				Mat maxv, maxi;
				Math::reduceMax<T>(weighted, maxv, maxi);

				// choose the best indices
				Mat Ixm, Iym;
				Math::reducePickIndex<int>(Ixp, maxi, Ixm);
				Math::reducePickIndex<int>(Iyp, maxi, Iym);
				Ix[n][c][p][m] = Ixm;
				Iy[n][c][p][m] = Iym;
				Ik[n][c][p][m] = maxi;

				// update the parent's score
				ComponentPart parent = cpart.parent();
				if (parent.score(ncscores,m).empty()) parent.score(scores[n],m).copyTo(parent.score(ncscores,m));
				parent.score(ncscores,m) += maxv;
				if (parent.self() == 0) {
					ComponentPart root = parts.component(c);
				}
			}
		}
		// add bias to the root score and find the best mixture
		ComponentPart root = parts.component(c);
		Mat rncscore = root.score(ncscores,0);
		T bias = root.bias(0)[0];
		vectorMat weighted;
		// weight each of the child scores
		for (unsigned int m = 0; m < root.nmixtures(); ++m) {
			weighted.push_back(root.score(ncscores,m) + bias);
		}
		Math::reduceMax<T>(weighted, rootv[n][c], rooti[n][c]);
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
void DynamicProgram<T>::argmin(Parts& parts, const vector2DMat& rootv, const vector2DMat& rooti, const vectorf scales, const vector4DMat& Ix, const vector4DMat& Iy, const vector4DMat& Ik, vectorCandidate& candidates) {

	// for each scale, and each component, traverse back down the tree to retrieve the part positions
	const unsigned int nscales = scales.size();
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (unsigned int n = 0; n < nscales; ++n) {
		T scale = scales[n];
		for (unsigned int c = 0; c < parts.ncomponents(); ++c) {

			// get the scores and indices for this tree of parts
			const vector2DMat& Iknc = Ik[n][c];
			const vector2DMat& Ixnc = Ix[n][c];
			const vector2DMat& Iync = Iy[n][c];
			const unsigned int nparts = parts.nparts(c);

			// threshold the root score
			Mat over_thresh = rootv[n][c] > thresh_;
			Mat rootmix     = rooti[n][c];
			vectorPoint inds;
			Math::find(over_thresh, inds);

			for (unsigned int i = 0; i < inds.size(); ++i) {
				Candidate candidate;
				candidate.setComponent(c);
				vectori     xv(nparts);
				vectori     yv(nparts);
				vectori     mv(nparts);
				for (unsigned int p = 0; p < nparts; ++p) {
					ComponentPart part = parts.component(c, p);
					// calculate the child's points from the parent's points
					unsigned int x, y, m;
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

