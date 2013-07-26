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

#ifndef SPATIAL_CONVOLUTION_ENGINE_HPP_
#define SPATIAL_CONVOLUTION_ENGINE_HPP_

#include "IConvolutionEngine.hpp"

class SpatialConvolutionEngine: public IConvolutionEngine {
private:
	//! the internally supported convolution type, taken from the filter type
	int type_;
	//! the number of layers to each filter
	size_t flen_;
	//! the internal representation of the filters
	vector2DFilterEngine filters_;
	void convolve(const cv::Mat& feature, vectorFilterEngine& filter, cv::Mat& pdf, const size_t stride);
public:
	SpatialConvolutionEngine(int type, size_t flen);
	virtual ~SpatialConvolutionEngine();
	virtual void setFilters(const vectorMat& filters);
	virtual void pdf(const vectorMat& features, vector2DMat& responses);
};

#endif /* SPATIAL_CONVOLUTION_ENGINE_HPP_ */
