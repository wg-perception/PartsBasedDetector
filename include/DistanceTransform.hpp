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
 *  File:    DistanceTransform.cpp
 *  Author:  Hilton Bristow
 *  Created: Aug 28, 2012
 */

#ifndef DISTANCETRANSFORM_HPP_
#define DISTANCETRANSFORM_HPP_

#include <limits>
#include <opencv2/core/core.hpp>

// ---------------------------------------------------------------------------
// SAMPLED FUNCTION INTERFACE
// ---------------------------------------------------------------------------

/*! @class PenaltyFunction
 *  @brief an interface for penalty functions to be passed to DistanceTransform
 *
 *  This penalty function interface defines the methods required by a function
 *  to be used in DistanceTransform. The function must be convex, so that the
 *  intersection between two such functions upholds a set of invariants.
 */
class PenaltyFunction {
public:
	/*! @brief intersection operator
	 *
	 * The intersection operator is used to find the height of intersection
	 * between two offset penalty functions.
	 *
	 * @param x0 the location of the first function
	 * @param x1 the location of the second function
	 * @param y0 the height of the underlying sampled function at x0
	 * @param y1 the height of the underlying sampled function at x1
	 * @return the height of the point of intersection
	 */
	virtual double operator() (const int x0, const int x1, const double y0, const double y1) const = 0;

	/*! @brief lower-envelope operator
	 *
	 * given the pixel location of the lower envelope, calculate the true
	 * function height at that point
	 * @param x the lower envelope's location on the function
	 * @param y the height of the underlying sampled point
	 * @return
	 */
	virtual double operator() (const int x, const double y) const = 0;
	virtual ~PenaltyFunction() {}
};

/*! @class Quadratic
 *  @brief quadratic penalty function
 *
 *  This class implements a Euclidean distance penalty function
 *  which manifests as a quadratic function
 */
class Quadratic : public PenaltyFunction {
private:
	int square(int x) const { return x*x; }
public:
	double const a;
	double const b;
	// constructor
	Quadratic(double _a, double _b) : a(_a), b(_b) {}
	// intersection operator
	double operator() (const int x0, const int x1, const double y0, const double y1) const {
		return ((y1-y0) - b*(x1-x0) + a*(square(x1) - square(x0))) / (2*a*(x1-x0));
	}
	// f(x) lower envelope operator
	double operator() (const int x, const double y) const {
		return a*square(x) + b*x + y;
	}
};

// ---------------------------------------------------------------------------
// DECLARATION
// ---------------------------------------------------------------------------

/*! @class DistanceTransform
 *
 *  @brief class for performing distance transforms of sampled functions
 *
 *  This distance transform can be used to reduce the complexity of algorithms
 *  such as dynamic programming, where the spatial distance penalty function
 *  is convex. This distance function is defined by the PenaltyFunction
 *  interface.
 *
 *  The distance transform is a separable operation, so a 2D distance transform
 *  will be applied first over the rows, then over the columns
 */
template<typename T>
class DistanceTransform {
private:
	inline void computeRow(T const * const src, T * const dst, int * const ptr, const size_t N, const PenaltyFunction& f, int os=0) const;
public:
	DistanceTransform() {}
	virtual ~DistanceTransform() {}
	void compute(const cv::Mat_<T>& score_in, const PenaltyFunction& fx, const PenaltyFunction& fy, const cv::Point os, cv::Mat_<T>& score_out, cv::Mat_<int>& Ix, cv::Mat_<int>& Iy) const;
};


// ---------------------------------------------------------------------------
// IMPLEMENTATION
// ---------------------------------------------------------------------------

/*! @brief Generalized 1D distance transform
 *
 * This method performs the 1D distance transform across the rows of a matrix.
 * It is called twice internally by distanceTransform(), once across the rows
 * and once down the columns (transposed to rows)
 *
 * @param src pointer to the start of the source data
 * @param dst pointer to the start of the destination data
 * @param ptr pointer to the indices
 * @param N the total number of rows
 * @param f the 1D distance penalty function
 * @param os the anchor offset
 */
template<typename T>
inline void DistanceTransform<T>::computeRow(T const * const src, T * const dst, int * const ptr, const size_t N, const PenaltyFunction& f, int os) const {

	int * const v = new int[N];
	T   * const z = new T[N+1];
	int k = 0;
	v[0] = 0;
	z[0] = -std::numeric_limits<T>::infinity();
	z[1] = +std::numeric_limits<T>::infinity();
	for (size_t q = 1; q < N; ++q) {
		T s = f(v[k], q, src[v[k]], src[q]);
		while (s <= z[k] && k > 0) {
			k--;
			s = f(v[k], q, src[v[k]], src[q]);
		}
		k++;
		v[k]   = q;
		z[k]   = s;
		z[k+1] = +std::numeric_limits<T>::infinity();
	}

	k = 0;
	for (size_t q = 0; q < N; ++q) {
		while (z[k+1] < os) k++;
		dst[q] = f(os-v[k], src[v[k]]);
		ptr[q] = v[k];
		os++;
	}

	delete [] v;
	delete [] z;
}

/*! @brief Generalized distance transform
 *
 * 2-Dimensional generalized distance transform based on the paper:
 * P. Felzenszwalb and D. Huttenlocher, "Distance Transforms of Sampled Functions,"
 * Cornell Technical Report, 2004
 *
 * This is used to reduce the complexity of the dynamic program, namely when all
 * of the cost functions are quadratic. The 2D distance transform is broken down
 * into two 1D transforms since the operation is separable
 *
 * @param score_in the input score
 * @param fx the distance penalty function in the x-dimension
 * @param fy the distance penalty function in the y-dimension
 * @param os the anchor offset of the child from the parent
 * @param score_out the distance transformed score
 * @param Ix the distances in the x direction
 * @param Iy the distances in the y direction
 */
template<typename T>
void DistanceTransform<T>::compute(const cv::Mat_<T>& score_in, const PenaltyFunction& fx, const PenaltyFunction& fy, const cv::Point os, cv::Mat_<T>& score_out, cv::Mat_<int>& Ix, cv::Mat_<int>& Iy) const {

	// get the dimensionality of the score
	const size_t M = score_in.rows;
	const size_t N = score_in.cols;

	// allocate the output and working matrices
	score_out.create(cv::Size(M, N));
	Ix.create(cv::Size(N, M));
	Iy.create(cv::Size(M, N));
	cv::Mat_<T> score_tmp(cv::Size(N, M));

	// compute the distance transform across the rows
	for (size_t m = 0; m < M; ++m) {
		computeRow(score_in[m], score_tmp[m], Ix[m], N, fx, os.x);
	}

	// transpose the intermediate matrices
	transpose(score_tmp, score_tmp);

	// compute the distance transform down the columns
	for (size_t n = 0; n < N; ++n) {
		computeRow(score_tmp[n], score_out[n], Iy[n], M, fy, os.y);
	}

	// transpose back to the original layout
	transpose(score_out, score_out);
	transpose(Iy, Iy);

	// get argmins
	cv::Mat_<int> row(cv::Size(N, 1));
	int * const row_ptr = row[0];
	for (size_t m = 0; m < M; ++m) {
		int * const Iy_ptr = Iy[m];
		int * const Ix_ptr = Ix[m];
		for (size_t n = 0; n < N; ++n) {
			row_ptr[n] = Iy_ptr[Ix_ptr[n]];
		}
		for (size_t n = 0; n < N; ++n) {
			Iy_ptr[n] = row_ptr[n];
		}
	}
}

#endif /* DISTANCETRANSFORM_HPP_ */
