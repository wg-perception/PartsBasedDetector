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
#include "ITree.hpp"
typedef std::vector<std::vector<float> > vector2Df;
typedef std::vector<std::vector<cv::Mat> > vector2DMat;

/*
 *
 */
class Part : public ITree<std::vector<cv::Mat> > {
private:
	// --------------------
	// Part members
	//! the part bias (recognition reliability)
	vector2Df bias_;
	//! the patch expert (SVM)
	std::vector<cv::Mat> filter_;
	//! the number of mixtures ( filter_.size() == nmixtures_ )
	int nmixtures_;
	//! the quadratic weights for each mixture
	vector2Df w_;
	//! the position of the part relative to its parent
	cv::Point anchor_;
	//! the linear Part position when indexing into vectors
	int pos_;
	// --------------------
	// ITree members
	//! the total number of children below this node
	int ndescendants_;
	//! the level of this node below the root
	int level_;
	//! the child parts
	std::vector<Part> children_;
public:
	Part() {}
	Part(vector2Df& bias, std::vector<cv::Mat>& filter, int pos, std::vector<Part>& children, int level, int ndescendants) :
		bias_(bias), filter_(filter), nmixtures_(filter.size()), pos_(pos),
		children_(children), level_(level), ndescendants_(ndescendants) {}
	virtual ~Part() {}
	// get methods (this is a constant class, so set methods are not allowed)
	const vector2Df& bias(void) const { return bias_; }
	const std::vector<cv::Mat> filter(void) const { return filter_; }
	virtual const std::vector<Part> children(void) const { return children_; }
	const int nmixtures(void) const { return nmixtures_; }
	const vector2Df w(void) const { return w_; }
	const cv::Point anchor(void) const { return anchor_; }
	const int pos(void) const { return pos_; }
	// ITree methods
	const int ndescendants(void) const { return ndescendants_; }
	const int level(void) const { return level_; }
	const std::vector<cv::Mat> value(void) const { return filter_; }
	const bool isLeaf(void) const { return ndescendants_ == 0; }

	static Part constructPartHierarchy(vector2DMat& filters, std::vector<int>& parents);
};

#endif /* PART_HPP_ */
