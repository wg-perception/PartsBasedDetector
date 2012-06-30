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
 *  File:    Visualize.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#include "Visualize.hpp"
using namespace cv;
using namespace std;

/*! @brief Visualize the candidate part locations overlaid on an image
 *
 * @param im the image
 * @param candidates a vector of type Candidate, representing potential
 * part locations
 */
void Visualize::candidates(const cv::Mat& im, std::vector<Candidate> candidates) {

	// create a new canvas that we can modify
	Mat canvas;
	im.copyTo(canvas);

	// generate a set of colors to display. Do this in HSV then convert it
	int ncolors = candidates[0].parts().size();
	vector<Scalar> colors;
	for (int n = 0; n < ncolors; ++n) {
		Mat_<int> color = 0.5*Mat::ones(Size(1,1), CV_32FC3);
		// Hue is in radians
		color(0,0,0) = (2 * CV_PI) / ncolors * n;
		cvtColor(color, color, CV_HSV2BGR);
		colors.push_back(Scalar(color(0,0,0), color(0,0,1), color(0,0,2)));
	}

	// draw each candidate to the canvas
	const int LINE_THICKNESS = 5;
	for (int n = 0; n < candidates.size(); ++n) {
		Candidate candidate = candidates[n];
		for (int p = 0; p < candidate.parts().size(); ++p) {
			rectangle(canvas, candidate.parts()[p], colors[p], LINE_THICKNESS);
		}
	}

	imshow(name_, canvas);
}
