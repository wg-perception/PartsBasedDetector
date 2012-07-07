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
 *  File:    PartsBasedDetector.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#include "PartsBasedDetector.hpp"
using namespace cv;
using namespace std;

PartsBasedDetector::PartsBasedDetector() {
	// TODO Auto-generated constructor stub

}

PartsBasedDetector::~PartsBasedDetector() {
	// TODO Auto-generated destructor stub
}

vector<Candidate> PartsBasedDetector::detect(const Mat& im) {

	// calculate a feature pyramid for the new image
	vector<Mat> pyramid;
	features_.pyramid(im, pyramid);

	// convolve the feature pyramid with the Part experts
	// to get probability density for each Part
	vector<Mat> filters;
	vector<Mat> pdf;
	root_.toVector(filters);
	features_.pdf(pyramid, filters, pdf);

	// use dynamic programming to predict the best detection candidates from the part responses
	Mat maxv, maxi;
	dp_.min(root_, pdf, features_.nscales(), maxv, maxi);

	// walk back down the tree to find the part locations
	vector<Candidate> candidates = dp_.argmin(root_, pdf, features_.nscales(), maxv, maxi);

	return candidates;
}

/*! @brief Distribute the model parameters to the PartsBasedDetector classes
 *
 * @param model the monolithic model containing the deserialization of all model parameters
 */
void PartsBasedDetector::distributeModel(Model& model) {

	// the name of the Part detector
	name_ = model.name();

	// initialize the tree of Parts
	root_ = Part::constructPartHierarchy(model.filters(), model.conn());

	// initialize the Feature engine
	features_ = HOGFeatures<float>(model.binsize(), model.nscales(), model.flen(), model.norient());

	// initialize the dynamic program
	dp_ = DynamicProgram(model.thresh());

}
