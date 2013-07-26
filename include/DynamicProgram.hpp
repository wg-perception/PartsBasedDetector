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
 *  File:    DynamicProgram.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef DYNAMICPROGRAM_HPP_
#define DYNAMICPROGRAM_HPP_
#include <vector>
#include <opencv2/core/core.hpp>
#include "Candidate.hpp"
#include "DistanceTransform.hpp"
#include "Model.hpp"
#include "Parts.hpp"
#include "types.hpp"


/*! @class DynamicProgram
 *  @brief Dynamic Program to calculate the best holistic detection
 *
 *  The Dynamic Program calculates the best holistic detection given a
 *  set of part detections and a model of the parts' likely relationships
 *  with their parents. The class has two primary methods: min() and argmin().
 *
 *  min() computes the best candidates by passing messages from the leaves
 *  of the Part tree to the root. argmin() traverses back down the tree to
 *  retrieve the actual Part locations
 */
template<typename T>
class DynamicProgram {
private:
	//! the threshold for a positive detection
	double thresh_;
	DistanceTransform<T> dt_;
	void distanceTransform1D(const T* src, T* dst, int* ptr, size_t n, T a, T b, int os);
	void distanceTransform1DMat(const cv::Mat_<T>& src, cv::Mat_<T>& dst, cv::Mat_<int>& ptr, size_t N, T a, T b, int os);
public:
	DynamicProgram() {}
	DynamicProgram(double thresh) : thresh_(thresh) {}
	virtual ~DynamicProgram() {}
	// public methods
	void min(Parts& parts, vector2DMat& scores, vector4DMat& Ix, vector4DMat& Iy, vector4DMat& Ik, vector2DMat& rootv, vector2DMat& rooti);
	void argmin(Parts& parts, const vector2DMat& rootv, const vector2DMat& rooti, const vectorf scales, const vector4DMat& Ix, const vector4DMat& Iy, const vector4DMat& Ik, vectorCandidate& candidates);
	void distanceTransform(const cv::Mat& score_in, const vectorf w, cv::Point os, cv::Mat& score_out, cv::Mat& Ix, cv::Mat& Iy);
};

#endif /* DYNAMICPROGRAM_HPP_ */
