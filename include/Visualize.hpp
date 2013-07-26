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
 *  File:    Visualize.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef VISUALIZE_HPP_
#define VISUALIZE_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include "Candidate.hpp"
#include "types.hpp"

/*! @class Visualize
 *  @brief visualize detection candidates
 *
 * visualize a collection of object detection candidates by rendering the
 * input image to screen, and overlaying the detection bounding boxes of
 * each of the parts, with optional confidence values
 */
class Visualize {
private:
	//! the name of the OpenCV window
	std::string name_;
public:
	Visualize() : name_("frame") {}
	Visualize(std::string name) : name_(name) {}
	virtual ~Visualize() {}
	// public methods
	void candidates(const cv::Mat& im, const vectorCandidate& candidates, cv::Mat& canvas, bool display_confidence = false) const;
	void candidates(const cv::Mat& im, const vectorCandidate& candidates, size_t N, cv::Mat& canvas, bool display_confidence = false) const;
	void candidates(const cv::Mat& im, const Candidate& candidate, cv::Mat& canvas, bool display_confidence = true) const;
	void image(const cv::Mat& im) const;
};

#endif /* VISUALIZE_HPP_ */
