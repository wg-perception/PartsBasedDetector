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
 *  File:    ModelTransfer.cpp
 *  Author:  Hilton Bristow
 *  Created: Jul 27, 2012
 */

#include <iostream>
#include "MatlabIOModel.hpp"
#include "FileStorageModel.hpp"
using namespace std;

int main(int argc, char** argv) {

	// check for usage
	if (argc != 3) {
		cerr << "Usage: ModelTransfer /path/to/mat/file /path/to/xml/file" << endl;
		exit(-1);
	}
	// allocate two models
	Model* matlab = new MatlabIOModel;
	Model* cv     = new FileStorageModel;

	// deserialize the Matlab model, cast sideways and serialize
	// an OpenCV FileStorage model
	cout << "-------------------------------" << endl;
	cout << "        Model Transfer         " << endl;
	cout << "-------------------------------" << endl;
	cout << "" << endl;
	cout << "deserializing Matlab (.mat) model..." << endl;
	matlab->deserialize(argv[1]);
	cout << "converting..." << endl;
	(*cv) = (*matlab);
	cout << "serializing to OpenCV (.xml) model..." << endl;
	cv->serialize(argv[2]);
	cout << "Conversion complete" << endl;
	cout << "-------------------------------" << endl;

	// cleanup
	delete matlab;
	delete cv;
	return 0;
}
