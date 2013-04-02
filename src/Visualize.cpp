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

#include <cstdio>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/lexical_cast.hpp>
#include "Visualize.hpp"
using namespace cv;
using namespace std;

/*! @brief visualize the candidate part locations overlaid on an image
 *
 * @param im the image
 * @param candidates a vector of type Candidate, representing potential
 * part locations
 * @param N the number of candidates to render. If the candidates have been sorted,
 * this is equivalent to displaying only the 'N best' candidates
 * @param display_confidence display the detection confidence above each bounding box
 * for each part
 */
void Visualize::candidates(const Mat& im, const vectorCandidate& candidates, unsigned int N, Mat& canvas, bool display_confidence) const {

	// create a new canvas that we can modify
    cvtColor(im, canvas, CV_RGB2BGR);
    if (candidates.size() == 0) return;

	// generate a set of colors to display. Do this in HSV then convert it
	const unsigned int ncolors = candidates[0].parts().size();
	vector<Scalar> colors;
	for (unsigned int n = 0; n < ncolors; ++n) {
		Mat color(Size(1,3), CV_32FC3);
		// Hue is in degrees, not radians (because consistency is over-rated)
		color.at<float>(0) = (360) / ncolors * n;
		color.at<float>(1) = 1.0;
		color.at<float>(2) = 0.7;
		cvtColor(color, color, CV_HSV2BGR);
		color = color * 255;
		colors.push_back(Scalar(color.at<float>(0), color.at<float>(1), color.at<float>(2)));
	}

	// draw each candidate to the canvas
	const int LINE_THICKNESS = 4;
	Scalar black(0,0,0);
	N = (candidates.size() < N) ? candidates.size() : N;
	for (unsigned int n = 0; n < N; ++n) {
		Candidate candidate = candidates[n];
		for (unsigned int p = 0; p < candidate.parts().size(); ++p) {
			Rect box = candidate.parts()[p];
			string confidence  = boost::lexical_cast<string>(candidate.confidence()[p]);
			rectangle(canvas, box, colors[p], LINE_THICKNESS);
			if (display_confidence && p == 0) putText(canvas, confidence, Point(box.x, box.y-5), FONT_HERSHEY_SIMPLEX, 0.5f, black, 2);
		}
		//rectangle(canvas, candidate.boundingBox(), Scalar(255, 0, 0), LINE_THICKNESS);
	}
}

/*! @brief visualize all of the candidate part locations overlaid on an image
 *
 * @param im the image
 * @param candidates a vector of type Candidate, representing potential
 * part locations
 * @param display_confidence display the detection confidence above each bounding box
 * for each part
 */
void Visualize::candidates(const Mat& im, const vector<Candidate>& candidates, Mat& canvas, bool display_confidence) const {
	Visualize::candidates(im, candidates, candidates.size(), canvas, display_confidence);
}

/*! @brief visualize a single candidate overlaid on an image
 *
 * @param im the image
 * @param candidate a single Candidate to superimpose over the image
 * @param display_confidence display the detection confidence above each bounding box
 * for each part
 */
void Visualize::candidates(const Mat& im, const Candidate& candidate, Mat& canvas, bool display_confidence) const {

	vector<Candidate> vec;
	vec.push_back(candidate);
	candidates(im, vec, canvas, display_confidence);
}

/*! @brief display the raw image with no overlay
 *
 * @param im the input image frame
 */
void Visualize::image(const Mat& im) const {
    Mat canvas;
    cvtColor(im, canvas, CV_RGB2BGR);
	namedWindow(name_, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	imshow(name_, canvas);
}
