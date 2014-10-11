/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Willow Garage, Inc.
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
 *  File:    SpatialConvolutionEngine.hpp
 *  Author:  Hilton Bristow
 *  Created: Oct 9, 2012 
 */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cassert>
#include "SpatialConvolutionEngine.hpp"
using namespace std;
using namespace cv;

SpatialConvolutionEngine::SpatialConvolutionEngine(int type, size_t flen) :
	type_(type), flen_(flen) {}

SpatialConvolutionEngine::~SpatialConvolutionEngine() {
	// TODO Auto-generated destructor stub
}

/*! @brief Convolve two matrices, with a stride of greater than one
 *
 * This is a specialized 2D convolution algorithm with a stride of greater
 * than one. It is designed to convolve a filter with a feature, where at
 * each pixel an SVM must be evaluated (leading to a stride of SVM weight length).
 * The convolution can be thought of as flattened a 2.5D convolution where the
 * (i,j) dimension is the spatial plane and the (k) dimension is the SVM weights
 * of the pixels.
 *
 * The function supports multithreading via OpenMP
 *
 * @param feature the feature matrix
 * @param filter the filter (SVM)
 * @param pdf the response to return
 * @param stride the SVM weight length
 */
void SpatialConvolutionEngine::convolve(const Mat& feature, vectorFilterEngine& filter, Mat& pdf, const size_t stride) {

	// error checking
	assert(feature.depth() == type_);

	// split the feature into separate channels
	vectorMat featurev;
	split(feature.reshape(stride), featurev);

	// calculate the output
	Rect roi(0,0,-1,-1); // full image
	Point offset(0,0);
	Size fsize = featurev[0].size();
	pdf = Mat::zeros(fsize, type_);

	for (size_t c = 0; c < stride; ++c) {
		Mat pdfc(fsize, type_);
		filter[c]->apply(featurev[c], pdfc, roi, offset, true);
		pdf += pdfc;
	}
}


/*! @brief Calculate the responses of a set of features to a set of filter experts
 *
 * A response represents the likelihood of the part appearing at each location of
 * the feature map. Parts are support vector machines (SVMs) represented as filters.
 * The convolution of a filter with a feature produces a probability density function
 * (pdf) of part location
 * @param features the input features (at different scales, and by extension, size)
 * @param responses the vector of responses (pdfs) to return
 */
void SpatialConvolutionEngine::pdf(const vectorMat& features, vector2DMat& responses) {

	// preallocate the output
	const size_t M = features.size();
	const size_t N = filters_.size();
	responses.resize(M, vectorMat(N));

	// iterate
#ifdef _OPENMP
	#pragma omp parallel for
#endif
	for (size_t n = 0; n < N; ++n) {
		for (size_t m = 0; m < M; ++m) {
			Mat response;
			convolve(features[m], filters_[n], response, flen_);
			responses[m][n] = response;
		}
	}
}

/*! @brief set the filters
 *
 * given a set of filters, split each filter channel into a plane,
 * in preparation for convolution
 *
 * @param filters the filters
 */
void SpatialConvolutionEngine::setFilters(const vectorMat& filters) {

	const size_t N = filters.size();
	filters_.clear();
	filters_.resize(N);

	// split each filter into separate channels, and create a filter engine
	const size_t C = flen_;
	for (size_t n = 0; n < N; ++n) {
		vectorMat filtervec;
		std::vector<Ptr<FilterEngine> > filter_engines(C);
		split(filters[n].reshape(C), filtervec);

		// the first N-1 filters have zero-padding
		for (size_t m = 0; m < C-1; ++m) {
			Ptr<FilterEngine> fe = createLinearFilter(type_, type_,
					filtervec[m], Point(-1,-1), 0, BORDER_CONSTANT, -1, Scalar(0,0,0,0));
			filter_engines[m] = fe;
		}

		// the last filter has one-padding
		Ptr<FilterEngine> fe = createLinearFilter(type_, type_,
				filtervec[C-1], Point(-1,-1), 0, BORDER_CONSTANT, -1, Scalar(1,1,1,1));
		filter_engines[C-1] = fe;
		filters_[n] = filter_engines;
	}
}
