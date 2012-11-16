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
#include <pcl/common/centroid.h>
#include "Node.hpp"
#include "PointCloudClusterer.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	// register the node
	ros::init(argc, argv, "object_recognition_by_parts",
			ros::init_options::AnonymousName);
	PartsBasedDetectorNode pbdn;
	bool ok = pbdn.init();
	if (!ok)
		exit(-1);
	ros::spin();
	return 0;
}

bool compareClusters(pcl::PointIndices first, pcl::PointIndices second)
{
	return first.indices.size() > second.indices.size();
}

bool PartsBasedDetectorNode::init(void)
{

	// attempt to load the model file and distribute the parameters
	string modelfile;

	// private nodehandle
	ros::NodeHandle priv_nh("~");
	priv_nh.getParam("model", modelfile);
  
	string ext = boost::filesystem::path(modelfile).extension().c_str();
	ROS_INFO("Loading model %s", modelfile.c_str());

	// OpenCV FileStorageModel
	if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0)
	{
		FileStorageModel model;
		bool ok = model.deserialize(modelfile);
		if (!ok)
		{
			fprintf(stderr, "Error deserializing file\n");
			return false;
		}
		pbd_.distributeModel(model);
		name_ = model.name();
	}
#ifdef WITH_MATLABIO
	// cvmatio MatlabIOModel
	else if (ext.compare(".mat") == 0)
	{
		MatlabIOModel model;
		bool ok = model.deserialize(modelfile);
		if (!ok)
		{
			fprintf(stderr, "Error deserializing file\n");
			return false;
		}
		pbd_.distributeModel(model);
		name_ = model.name();
	}
#endif
	else
	{
		fprintf(stderr, "Unsupported model format: %s\n", ext.c_str());
		return false;
	}

	// setup the detector publishers
	// register the callback for synchronised depth and camera images
	sync_.registerCallback(
			boost::bind(&PartsBasedDetectorNode::detectorCallback, this, _1, _2,
					_3));
	info_sub_d_.registerCallback(&PartsBasedDetectorNode::depthCameraCallback,
			this);

	// initialise the publishers
//	image_pub_d_ = it_.advertise(ns_ + name_ + "/depth_rect", 1);
	image_pub_rgb_ = it_.advertise(ns_ + name_ + "/candidates_rect_color", 1);
	mask_pub_ = it_.advertise(ns_ + name_ + "/mask", 1);
//	segmented_pub_ = it_.advertise(ns_ + name_ + "/segmented_image", 1);
	bb_pub_ = nh_.advertise<MarkerArray>(ns_ + name_ + "/bounding_box", 1);
	cloud_pub_ = nh_.advertise<PointCloud>(ns_ + name_ + "/cleaned_cloud", 1);
	part_center_pub_ = nh_.advertise<MarkerArray>(ns_ + name_ + "/part_centers",
			1);
	object_pose_pub_ = nh_.advertise<PoseArray>(ns_ + name_ + "/object_poses",
			1);

	// if we got here, everything is okay
	return true;
}

void PartsBasedDetectorNode::depthCameraCallback(
		const CameraInfoConstPtr& info_msg)
{
	depth_camera_ = *info_msg;
	depth_camera_initialized_ = true;
}

void PartsBasedDetectorNode::detectorCallback(const ImageConstPtr& msg_d,
		const ImageConstPtr& msg_rgb, const PointCloud::ConstPtr& msg_cloud)
{

	typedef PointCloudClusterer<PointType> PointCloudClusterer;

	// UNPACK PREAMBLE
	// update the stereo camera parameters from the depth and rgb camera info
	if (!depth_camera_initialized_)
		return;
	camera_.fromCameraInfo(depth_camera_);

	// convert the ROS image payloads to OpenCV structures
	cv_bridge::CvImagePtr cv_ptr_d;
	cv_bridge::CvImagePtr cv_ptr_rgb;
	try
	{
		cv_ptr_d = cv_bridge::toCvCopy(msg_d, enc::TYPE_32FC1);
		cv_ptr_rgb = cv_bridge::toCvCopy(msg_rgb, enc::BGR8);
	} catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s\n", e.what());
		return;
	}

	// strip out the matrices
	Mat image_d = cv_ptr_d->image;
	Mat image_rgb = cv_ptr_rgb->image;

	// DETECT
	vectorCandidate candidates;
	pbd_.detect(image_rgb, image_d, candidates);

	if (candidates.size() == 0)
		return;

	// PUBLISH
	// perform non-maximal suppression
	if (candidates.size() > 1)
	{
		Candidate::sort(candidates);
		Candidate::nonMaximaSuppression(image_rgb, candidates, 0.1); // 10% overlap allowed
	}

	ROS_INFO("Found %zu candidates.", candidates.size());

	//pose calculation members
	std::vector<Rect3d> bounding_boxes;
	std::vector<float> scores;
	std::vector<PointCloud> part_centers;
	std::vector<PointCloud> clusters;
	std::vector<PointType> object_centers;

	if (bb_pub_.getNumSubscribers() > 0 || cloud_pub_.getNumSubscribers() > 0
			|| part_center_pub_.getNumSubscribers() > 0
			|| object_pose_pub_.getNumSubscribers() > 0)
	{

		PointCloudClusterer::PointProjectFunc projecter = boost::bind(&PinholeCameraModel::projectPixelTo3dRay, &camera_, _1);
		PointCloudClusterer::computeBoundingBoxes(candidates, image_rgb,
				image_d, projecter, msg_cloud, bounding_boxes, part_centers);
	}

	if (cloud_pub_.getNumSubscribers() > 0
			|| object_pose_pub_.getNumSubscribers() > 0)
	{
		PointCloudClusterer::clusterObjects(msg_cloud, bounding_boxes, clusters,
				object_centers);
	}

	// publish on the various topics (only if there are subscribers)
//	if (image_pub_d_.getNumSubscribers() > 0)
//		messageImageDepth(image_d, msg_d);
	if (image_pub_rgb_.getNumSubscribers() > 0)
		messageImageRGB(candidates, image_rgb, msg_d);
	if (bb_pub_.getNumSubscribers() > 0)
		messageBoundingBox(bounding_boxes, msg_d);
	if (mask_pub_.getNumSubscribers() > 0)
		messageMask(candidates, image_rgb, msg_rgb);
	if (cloud_pub_.getNumSubscribers() > 0)
	{
		PointCloud all_clusters = clusters[0]; //there is at least one candidate
		for(size_t i = 1; i < clusters.size(); ++i)
		{
			all_clusters += clusters[i];
		}

		cloud_pub_.publish(all_clusters);
	}
	if (object_pose_pub_.getNumSubscribers() > 0)
	{
		// Pose from part centers
		PoseArray poses;
		poses.header = msg_cloud->header;
		poses.poses.clear();

		// for each object
		for (int object_it = 0; object_it < part_centers.size(); ++object_it)
		{
			const PointCloud& part_centers_cloud = part_centers[object_it];

			//compute centroid
			Eigen::Vector4f centroid;
			Eigen::Matrix3f covMat;

			int point_count = 0;
			if ((point_count = computeMeanAndCovarianceMatrix(
					part_centers_cloud, covMat, centroid)) == 0)
			{
				ROS_WARN("Centroid not found...");
				continue;
			}

			// normalize matrix
			covMat /= point_count;

			//eigen33 -> RF
			Eigen::Matrix3f evecs;
			Eigen::Vector3f evals;
			pcl::eigen33(covMat, evecs, evals);
			Eigen::Quaternion<float> quat(evecs);
			quat.normalize();

			Pose pose;
			pose.position.x = centroid.x();
			pose.position.y = centroid.y();
			pose.position.z = centroid.z();

			pose.orientation.w = quat.w();
			pose.orientation.x = quat.x();
			pose.orientation.y = quat.y();
			pose.orientation.z = quat.z();

			poses.poses.push_back(pose);
		}

		object_pose_pub_.publish(poses);
	}
}
