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
 *  File:    FourierConvolutionEngine.hpp
 *  Author:  Hilton Bristow
 *  Created: July 6, 2013 
 */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "FourierConvolutionEngine.hpp"
using namespace std;
using namespace cv;

FourierConvolutionEngine::FourierConvolutionEngine(const Size& size, int type, size_t flen) :
	size_(Size(getOptimalDFTSize(size.width), getOptimalDFTSize(size.height))), type_(type), flen_(flen) {}

FourierConvolutionEngine::~FourierConvolutionEngine() {
	// TODO Auto-generated destructor stub
}

void FourierConvolutionEngine::convolve(const Mat& feature, vectorMat& filter, Mat& pdf, const size_t channels) {

  // error checking
  assert(feature.depth() == type_);

  // split the feature into separate channels
  vectorMat featurevec;
  split(feature.reshape(channels), featurevec);
  Size size = featurevec[0].size();
  Rect valid(0, 0, size.width, size.height);

  // for each channel, convolve
  Mat temp = Mat::zeros(size_, type_);
  for (size_t c = 0; c < channels; ++c) {
    Mat padded = Mat::zeros(size_, type_);
    Mat corner(padded, valid);
    featurevec[c].copyTo(corner);
    dft(padded, padded, 0, size.height);
    mulSpectrums(padded, filter[c], padded, 0);
    temp += padded;
  }
  dft(temp, temp, DFT_INVERSE + DFT_SCALE, size.height);
  temp(valid).copyTo(pdf);
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
void FourierConvolutionEngine::pdf(const vectorMat& features, vector2DMat& responses) {
  
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
void FourierConvolutionEngine::setFilters(const vectorMat& filters) {

  // allocate space in the vector for the filters
  const size_t N = filters.size();
  filters_.clear();
  filters_.resize(N);

  // iterate over the filters
  const size_t C = flen_;
  for (size_t n = 0; n < N; ++n) {
    vectorMat filtervec(C);
    split(filters[n].reshape(C), filtervec);
   
    // for each channel, pad the filter to the optimal size and take the fourier transform
    for (size_t c = 0; c < C; ++c) {
      Mat padded = Mat::zeros(size_, type_);
      Mat corner(padded, Rect(0, 0, filtervec[c].cols, filtervec[c].rows));
      filtervec[c].copyTo(corner);
      dft(padded, filtervec[c], 0, filtervec[c].rows);
    }
  }
}
