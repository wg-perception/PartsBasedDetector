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
 *  File:    Math.hpp
 *  Author:  Hilton Bristow
 *  Created: Aug 3, 2012
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include <cassert>
#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "types.hpp"
/*
 *
 */
class Math {
private:
	Math() {}
public:
	virtual ~Math() {}

	/*! @brief return the median value of a matrix
	 *
	 * @param mat the input matrix
	 * @return the median value, of the same precision as the input
	 */
	template<typename T>
	static T median(const cv::Mat& mat) {
		cv::Mat scratch;
		mat.copyTo(scratch);
		cv::MatIterator_<T> first = scratch.begin<T>();
		cv::MatIterator_<T> last  = scratch.end<T>();
		cv::MatIterator_<T> middle = first + std::distance(first, last)/2;
		std::nth_element(first, middle, last);
		std::cerr << scratch << std::endl;
		return *middle;
	}


	/*! @brief find nonzero elements in a matrix
	 *
	 * Find all nonzero elements in a matrix of type CV_8U, and return
	 * the indieces (x,y) of the nonzero pixels in an array of Point
	 *
	 * @param binary the input image (usually the output of a comparison
	 * operation such as compare(), >, ==, etc)
	 * @param idx the output vector of nonzero indices
	 */
	static void find(const cv::Mat& binary, std::vector<cv::Point>& idx) {

		assert(binary.depth() == CV_8U);
		const size_t M = binary.rows;
		const size_t N = binary.cols;
		for (size_t m = 0; m < M; ++m) {
			const unsigned char* bin_ptr = binary.ptr<unsigned char>(m);
			for (size_t n = 0; n < N; ++n) if (bin_ptr[n] > 0) idx.push_back(cv::Point(n,m));
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
	template<typename T>
	static void reducePickIndex(const vectorMat& in, const cv::Mat& idx, cv::Mat& out) {

		// error checking
		const size_t K = in.size();
		if (K == 1) { in[0].copyTo(out); return; }
		double minv, maxv;
		minMaxLoc(idx, &minv, &maxv);
		assert(minv >= 0 && maxv < K);
		for (size_t k = 0; k < K; ++k) assert(in[k].size() == idx.size());

		// allocate the output array
		out.create(in[0].size(), in[0].type());

		// perform the indexing
		size_t M = in[0].rows;
		size_t N = in[0].cols;
		std::vector<const T*> in_ptr(K);
		if (in[0].isContinuous()) { N = M*N; M = 1; }
		for (size_t m = 0; m < M; ++m) {
			T* out_ptr = out.ptr<T>(m);
			const int*   idx_ptr = idx.ptr<int>(m);
			for (size_t k = 0; k < K; ++k) in_ptr[k] = in[k].ptr<T>(m);
			for (size_t n = 0; n < N; ++n) {
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
	static void reduceMax(const vectorMat& in, cv::Mat& maxv, cv::Mat& maxi) {

		// TODO: flatten the input into a multi-channel matrix for faster indexing
		// error checking
		const size_t K = in.size();
		if (K == 1) {
			// just return
			in[0].copyTo(maxv);
			maxi = cv::Mat::zeros(in[0].size(), cv::DataType<int>::type);
			return;
		}

		assert (K > 1);
		for (size_t k = 1; k < K; ++k) assert(in[k].size() == in[k-1].size());

		// allocate the output matrices
		maxv.create(in[0].size(), in[0].type());
		maxi.create(in[0].size(), cv::DataType<int>::type);

		size_t M = in[0].rows;
		size_t N = in[0].cols;

		std::vector<const T*> in_ptr(K);
		if (in[0].isContinuous()) { N = M*N; M = 1; }
		for (size_t m = 0; m < M; ++m) {
			T* maxv_ptr = maxv.ptr<T>(m);
			int* maxi_ptr = maxi.ptr<int>(m);
			for (size_t k = 0; k < K; ++k) in_ptr[k] = in[k].ptr<T>(m);
			for (size_t n = 0; n < N; ++n) {
				T v = -std::numeric_limits<T>::infinity();
				int i = 0;
				for (size_t k = 0; k < K; ++k) if (in_ptr[k][n] > v) { i = k; v = in_ptr[k][n]; }
				maxi_ptr[n] = i;
				maxv_ptr[n] = v;
			}
		}
	}

};


#endif /* MATH_HPP_ */
