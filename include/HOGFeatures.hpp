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
 *  File:    HOGFeatures.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef HOGFEATURES_HPP_
#define HOGFEATURES_HPP_
#include <vector>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "IFeatures.hpp"
#include "types.hpp"

/*! @class HOGFeatures
 *  @brief Implementation of IFeatures interface using HOG
 */
template<typename T>
class HOGFeatures : public IFeatures {
private:
	//! the spatial binning size
	size_t binsize_;
	//! the number of scales to compute features at
	size_t nscales_;
	//! the length of the feature at each bin (histogram size)
	size_t flen_;
	//! the number of orientations to bin
	size_t norient_;
	//! the scales of the features
	vectorf scales_;
	//! the scaling factor between successive levels in the pyramid
	float sfactor_;
	//! the interval between half resolution scales
	size_t interval_;

	// private methods
	void boundaryOcclusionFeature(cv::Mat& feature, const int flen, const int padsize);
	template<typename IT> void features(const cv::Mat& im, cv::Mat& feature) const;
public:
	HOGFeatures() {}
	HOGFeatures(size_t binsize, size_t nscales, size_t flen, size_t norient) :
		binsize_(binsize), nscales_(nscales), flen_(flen), norient_(norient) {
		// TODO: don't hard code this. Compute more intuitively from scales rather than interval
		interval_ = nscales_;
		sfactor_  = pow(2.0f, 1.0f/(float)interval_);
		assert(norient_%2 == 0);

	}
	virtual ~HOGFeatures() {}
	// get methods
	size_t binsize(void) const { return binsize_; }
	size_t nscales(void) const { return nscales_; }
	vectorf scales(void) const { return scales_; }
	void pyramid(const cv::Mat& im, vectorMat& pyrafeatures);
};

#endif /* HOGFEATURES_HPP_ */
