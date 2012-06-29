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
 *  File:    Part.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 28, 2012
 */

#ifndef PART_HPP_
#define PART_HPP_

#include <vector>
#include <opencv2/core/core.hpp>
typedef std::vector<std::vector<float> > vector2Df;

/*
 *
 */
class Part {
private:
	//! the part bias (recognition reliability)
	vector2Df bias_;
	//! the patch expert (SVM)
	std::vector<cv::Mat> filter_;
	//! the parent Part
	Part& parent_;
	//! the number of mixtures ( filter_.size() == nmixtures_ )
	int nmixtures_;
	//! the quadratic weights for each mixture
	vector2Df w_;
	//! the position of the part relative to its parent
	cv::Point anchor_;
public:
	Part(float bias, std::vector<cv::Mat> filter, Part parent) :
		bias_(bias), filter_(filter), parent_(parent), nmixtures_(filter.size()) {}
	virtual ~Part() {}
	// get methods (this is a constant class, so set methods are not allowed)
	const vector2Df& bias(void) const { return bias_; }
	const std::vector<cv::Mat>& filter(void) const { return filter_; }
	const Part& parent(void) const { return parent_; }
	const int nmixtures(void) const { return nmixtures_; }
	const vector2Df& w(void) const { return w_; }
	const cv::Point anchor(void) const { return anchor_; }
};

#endif /* PART_HPP_ */
