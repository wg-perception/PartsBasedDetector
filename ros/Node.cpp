/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
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
 */
#include <cstdio>
#include "Node.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	// register the node
	ros::init(argc, argv, "object_recognition_by_parts", ros::init_options::AnonymousName);
	PartsBasedDetectorNode pbdn;
	bool ok = pbdn.init();
	if (!ok) exit(-1);
	ros::spin();
	return 0;
}


bool PartsBasedDetectorNode::init(void) {

	// attempt to load the model file and distribute the parameters
	string modelfile;
	nh_.getParam("model", modelfile);
	string ext = boost::filesystem::path(modelfile).extension().c_str();

	// OpenCV FileStorageModel
	if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0) {
		FileStorageModel model;
		bool ok = model.deserialize(modelfile);
		if (!ok) { fprintf(stderr, "Error deserializing file\n"); return false; }
		pbd_.distributeModel(model);
		name_ = model.name();
	}
#ifdef WITH_MATLABIO
	// cvmatio MatlabIOModel
	else if (ext.compare(".mat") == 0) {
		MatlabIOModel model;
		bool ok = model.deserialize(modelfile);
		if (!ok) { fprintf(stderr, "Error deserializing file\n"); return false; }
		pbd_.distributeModel(model);
		name_ = model.name();
	}
#endif
	else {
		fprintf(stderr, "Unsupported model format: %s\n", ext.c_str());
		return false;
	}

	// setup the detector publishers
	// register the callback for synchronised depth and camera images
	sync_.registerCallback( boost::bind(&PartsBasedDetectorNode::detectorCallback, this, _1, _2 ) );
	info_sub_d_.registerCallback(&PartsBasedDetectorNode::depthCameraCallback, this);

    // initialise the publishers
	image_pub_d_   = it_.advertise(ns_ + name_ + "/depth_rect", 1);
    image_pub_rgb_ = it_.advertise(ns_ + name_ + "/candidates_rect_color", 1);
    mask_pub_      = it_.advertise(ns_ + name_ + "/mask", 1);
    bb_pub_        = nh_.advertise<MarkerArray>(ns_ + name_ + "/bounding_box", 1);

	// if we got here, everything is okay
	return true;
}

void PartsBasedDetectorNode::depthCameraCallback(const CameraInfoConstPtr& info_msg) {
	depth_camera_ = *info_msg;
	depth_camera_initialized_ = true;
}

void PartsBasedDetectorNode::detectorCallback(const ImageConstPtr& msg_d, const ImageConstPtr& msg_rgb) {

	// UNPACK PREAMBLE
    // update the stereo camera parameters from the depth and rgb camera info
	if (!depth_camera_initialized_) return;
	camera_.fromCameraInfo(depth_camera_);

    // convert the ROS image payloads to OpenCV structures
    cv_bridge::CvImagePtr cv_ptr_d;
    cv_bridge::CvImagePtr cv_ptr_rgb;
    try {
        cv_ptr_d   = cv_bridge::toCvCopy(msg_d, enc::TYPE_32FC1);
        cv_ptr_rgb = cv_bridge::toCvCopy(msg_rgb, enc::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s\n", e.what());
        return;
    }

    // strip out the matrices
    Mat image_d   = cv_ptr_d->image;
    Mat image_rgb = cv_ptr_rgb->image;

    // DETECT
    vectorCandidate candidates;
    pbd_.detect(image_rgb, image_d, candidates);

    // PUBLISH
	// perform non-maximal suppression
    if (candidates.size() > 0) {
    	Candidate::sort(candidates);
    	Candidate::nonMaximaSuppression(image_rgb, candidates, 0.2);
    }

	// publish on the various topics (only if there are subscribers)
	if (image_pub_d_.getNumSubscribers() > 0)	messageImageDepth(image_d, msg_d);
	if (image_pub_rgb_.getNumSubscribers() > 0)	messageImageRGB(candidates, image_rgb, msg_d);
	if (bb_pub_.getNumSubscribers() > 0)  		messageBoundingBox(candidates, image_rgb, image_d, msg_d, camera_);
	if (mask_pub_.getNumSubscribers() > 0)	    messageMask(candidates, image_rgb, msg_rgb);
 }
