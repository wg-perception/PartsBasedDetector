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
 *  File:    Messages.cpp
 *  Author:  Hilton Bristow
 *  Created: Sep 10, 2012
 */

#ifndef BOUNDINGBOX_HPP_
#define BOUNDINGBOX_HPP_

#include <boost/functional/hash.hpp>
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include "Candidate.hpp"
#include "types.hpp"
#include "Visualize.hpp"
#include "Node.hpp"

using namespace cv;
using namespace std;

void hashStringToColor(const string& str, Scalar& rgb) {

	boost::hash<string> hash;
	Mat color(Size(1,1), CV_32FC3);
	// Hue is in degrees, not radians (because consistency is over-rated)
	color.at<float>(0) = hash(str) % 360;
	color.at<float>(1) = 1.0;
	color.at<float>(2) = 0.7;
	cvtColor(color, color, CV_HSV2BGR);
	color = color * 255;
	rgb = Scalar(color.at<float>(0), color.at<float>(1), color.at<float>(2));
}

void PartsBasedDetectorNode::clearMarkerArray(MarkerArray& markers, ros::Publisher& publisher) {
	for (unsigned int n = 0; n < markers.markers.size(); ++n) {
		markers.markers[n].action = Marker::DELETE;
	}
	publisher.publish(markers);
	markers.markers.clear();
}

void PartsBasedDetectorNode::messageBoundingBox(const vectorCandidate& candidates, Mat& rgb, Mat& depth, const ImageConstPtr& msg_in, const PinholeCameraModel& camera) {

	// always clear the marker array since we assume at each time step
	// that there is no correspondence between the markers
	clearMarkerArray(bb_markers_, bb_pub_);

	// allocate the MarkerArray (just a vector)
	const unsigned int N = candidates.size();
	for (unsigned int n = 0; n < N; ++n) {

		// generate the bounding box
		Point3d tl, br;
		Rect3d cube = candidates[n].boundingBox3D(rgb, depth);
		tl = camera_.projectPixelTo3dRay(Point2d(cube.tl().x, cube.tl().y))*cube.z;
		br = camera_.projectPixelTo3dRay(Point2d(cube.br().x, cube.br().y))*(cube.z+cube.depth);

		// add each bounding box
		Marker bb;
		bb.type = Marker::CUBE;
		// copy out of image message
		bb.header.frame_id = msg_in->header.frame_id;
		// when the image came came in
		bb.header.stamp = msg_in->header.stamp;

		// set the pose of the bounding box
		bb.pose.position.x = (tl.x + br.x)/2;
		bb.pose.position.y = (tl.y + br.y)/2;
		bb.pose.position.z = (tl.z + br.z)/2;
		bb.pose.orientation.x = 0.0;
		bb.pose.orientation.y = 0.0;
		bb.pose.orientation.z = 0.0;
		bb.pose.orientation.w = 1.0;

		// set the size of the bounding box
		bb.scale.x = br.x - tl.x;
		bb.scale.y = br.y - tl.y;
		bb.scale.z = br.z - tl.z;

		// set the color
		Scalar color;
		hashStringToColor(name_, color);
		bb.color.r = color[0];
		bb.color.g = color[1];
		bb.color.b = color[2];
		bb.color.a = 0.5f;

		// flag the marker to be added
		bb.action   = Marker::ADD;
		bb.lifetime = ros::Duration(5.0); // seconds
		bb.id       = n;

		// push the Marker on to the list
		bb_markers_.markers.push_back(bb);
	}
	// publish the marker array
	bb_pub_.publish(bb_markers_);
}

void PartsBasedDetectorNode::messageFrustum(const vectorCandidate& candidates) {

}

void PartsBasedDetectorNode::messageImageRGB(const vectorCandidate& candidates, Mat& rgb, const ImageConstPtr& msg_in) {

	// overlay the detections on the image
	Visualize visualize;
	Mat canvas;
	cv_bridge::CvImage container;
	visualize.candidates(rgb, candidates, canvas, true);
	container.image = canvas;
	container.encoding = enc::RGB8;
	ImagePtr msg_out = container.toImageMsg();
	msg_out->header.frame_id = msg_in->header.frame_id;
	msg_out->header.stamp    = msg_in->header.stamp;
	image_pub_rgb_.publish(msg_out);
}

void PartsBasedDetectorNode::messageImageDepth(Mat& depth, const ImageConstPtr& msg_in) {

	// simply republish the depth image (for now)
	image_pub_d_.publish(msg_in);
}

void PartsBasedDetectorNode::messageMask(const vectorCandidate& candidates, Mat& rgb, const ImageConstPtr& msg_in) {

	Mat mask;
	cv_bridge::CvImage container;
	Candidate::mask(rgb, candidates, mask);
	container.image = mask;
	container.encoding = enc::MONO8;
	ImagePtr msg_out = container.toImageMsg();
	msg_out->header.frame_id = msg_in->header.frame_id;
	msg_out->header.stamp    = msg_in->header.stamp;
	mask_pub_.publish(msg_out);
}


#endif /* BOUNDINGBOX_HPP_ */
