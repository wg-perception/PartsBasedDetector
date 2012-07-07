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
 *  File:    Model.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef MODEL_HPP_
#define MODEL_HPP_
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
typedef std::vector<std::vector<cv::Mat> > vector2DMat;
typedef std::vector<std::vector<float> >   vector2Df;
/*
 *
 */
class Model {
protected:
	// member variables
	//! the filters (\a nparts_ * \a nmixtures_)
	vector2DMat filters_;
	//! the bias of each Part
	vector2Df bias_;
	//! a unique string identifier for the model
	std::string name_;
	//! the connectivity of the parts, where each element is a reference to the part's parent
	std::vector<int> conn_;
	//! the number of parts
	int nparts_;
	//! the number of mixtures
	int nmixtures_;
	//! the number of scales at which to compute features
	int nscales_;
	//! the threshold for a positive detection
	float thresh_;
	//! the spatial pooling size when computing features
	int binsize_;
	//! the length of the feature vector in each bin
	int flen_;
	//! the number of orientations pin HOG feature bin
	int norient_;

public:
	Model();
	virtual ~Model();
	vector2DMat& filters(void) { return filters_; }
	vector2Df& bias(void) { return bias_; }
	std::string name(void) const { return name_; }
	std::vector<int>& conn(void) { return conn_; }
	int nparts(void) const { return nparts_; }
	int nmixtures(void) const { return nmixtures_; }
	float thresh(void) const { return thresh_; }
	int binsize(void) const { return binsize_; }
	int nscales(void) const { return nscales_; }
	int flen(void) const { return flen_; }
	int norient(void) const { return norient_; }
};

#endif /* MODEL_HPP_ */
