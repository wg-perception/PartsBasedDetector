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
 *  File:    main.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 27, 2012
 */

#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "MatlabIOModel.hpp"
#include "Visualize.hpp"
#include "types.hpp"
using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	// check arguments
	if (argc != 3) printf("Usage: PartsBasedDetector model_file image_file\n");

	// create the model object and deserialize it
	MatlabIOModel model;
	model.deserialize(argv[1]);

	// create the PartsBasedDetector and distribute the model parameters
	PartsBasedDetector<float> pbd;
	pbd.distributeModel(model);

	// load the image from file
	Mat im = imread(argv[2]);

	// detect potential candidates in the image
	double t = (double)getTickCount();
	vector<Candidate> candidates;
	pbd.detect(im, candidates);
	printf("Detection time: %f\n", ((double)getTickCount() - t)/getTickFrequency());
	printf("Number of candidates: %d\n", candidates.size());

	// display the best candidates
	Visualize visualize(model.name());
	if (candidates.size() > 0) {
		Candidate::sort(candidates);
		visualize.candidates(im, candidates, 1, true);
		waitKey();
	}

	return 0;
}
