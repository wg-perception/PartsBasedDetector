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
#include <pcl/common/centroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
//#include <pcl/segmentation/region_growing.h>
//#include <pcl17/segmentation/impl/region_growing.hpp>
//#include <pcl/segmentation/region_growing_rgb.h>
//#include <pcl17/segmentation/impl/region_growing_rgb.hpp>
//#include <pcl17/segmentation/seeded_hue_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
//#include <pcl/segmentation/extract_labeled_clusters.h>
//#include <pcl/segmentation/min_cut_segmentation.h>
//#include <pcl/segmentation/sac_segmentation.h>
//#include <pcl17/segmentation/impl/min_cut_segmentation.hpp>
//#include <pcl17/sample_consensus/sac_model_plane.h>
//#include <pcl17/features/normal_3d_omp.h>
//#include <pcl17/filters/statistical_outlier_removal.h>
//#include <pcl17/search/organized.h>

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

//	name_ = "Coffee";	//TODO REMOVE

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
//	if (candidates.size() > 0) {
	Candidate::sort(candidates);
	Candidate::nonMaximaSuppression(image_rgb, candidates, 0.1); // 10% overlap allowed
//	}

	ROS_INFO("Found %zu candidates.", candidates.size());

	//pose calculation members
	std::vector<Rect3d> bounding_boxes;
	std::vector<float> scores;
	std::vector<PointCloud> part_centers;
	std::vector<PointCloud> clusters;
	std::vector<PointType> object_centers;
	//TODO add other conditions
	if (bb_pub_.getNumSubscribers() > 0 || cloud_pub_.getNumSubscribers() > 0
			|| part_center_pub_.getNumSubscribers() > 0
			|| object_pose_pub_.getNumSubscribers() > 0)
	{
		clearMarkerArray(part_center_markers_, part_center_pub_);

    PointCloudClusterer::PointProjectFunc projecter = boost::bind(&PinholeCameraModel::projectPixelTo3dRay, &camera_, _1);
		PointCloudClusterer::computeBoundingBoxes(candidates, image_rgb,
				image_d, projecter, msg_cloud, bounding_boxes, part_centers);

//		computeBoundingBoxes(candidates, image_rgb, image_d, *msg_cloud,
//				bounding_boxes, scores, part_center_markers_);
		//TODO add header to markers
	}

	ROS_INFO("OK");

	if (cloud_pub_.getNumSubscribers() > 0
			|| object_pose_pub_.getNumSubscribers() > 0)
	{
//		cleanCloud(msg_cloud, bounding_boxes, cleaned_cloud, poses);
//		ROS_INFO("The cleaned cloud has %zu points.", cleaned_cloud.size());

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
//	if (part_center_pub_.getNumSubscribers() > 0)
//	{
//		//add header to markers
//		for (size_t i = 0; i < part_center_markers_.markers.size(); ++i)
//		{
//			part_center_markers_.markers[i].header.stamp = msg_d->header.stamp;
//			part_center_markers_.markers[i].header.frame_id =
//					msg_d->header.frame_id;
//		}
//
//		part_center_pub_.publish(part_center_markers_);
//	}
}

void PartsBasedDetectorNode::computeBoundingBoxes(
		const vectorCandidate& candidates, const cv::Mat& rgb,
		const cv::Mat& depth, const PointCloud& cloud,
		std::vector<Rect3d>& bounding_boxes, std::vector<float>& scores,
		MarkerArray& parts_centers)
{
	bounding_boxes.clear();
	scores.clear();
	cv::Mat segmented_mask(rgb.size(), CV_8UC1, cv::Scalar(0));
	std::vector<int> part_cloud_indices;
	pcl::PointIndices object_indices;
	PointCloud::Ptr foreground_pts(new PointCloud());

	for (size_t i = 0; i < candidates.size(); ++i)
	{
		const Candidate& candidate = candidates[i];
		if (candidate.score() > 1.0)
			continue;

		const Rect3d cube = candidate.boundingBox3D(rgb, depth);
		Point3d tl, br;

		if (isnan(cube.x) || isnan(cube.y) || isnan(cube.z))
		{
			ROS_INFO("Candidate %zu bounding box contains nans.", i);
			continue;
		}

		ROS_INFO(
				"Score: %f - Cube data: %f %f %f %f %f %f", candidate.score(), cube.tl().x, cube.tl().y, cube.z, cube.br().x, cube.br().y, cube.depth);

		Marker pc;
		pc.type = Marker::POINTS;
		pc.id = i;

		//pose
		pc.pose.position.x = 0;
		pc.pose.position.y = 0;
		pc.pose.position.z = 0;
		pc.pose.orientation.w = 1;
		pc.pose.orientation.x = 0;
		pc.pose.orientation.y = 0;
		pc.pose.orientation.z = 0;

		//size of each point
		pc.scale.x = 0.03;
		pc.scale.y = 0.03;
		pc.scale.z = 0.03;

		// set the color
		Scalar color;
		hashStringToColor("candidate" + i, color);
		pc.color.r = color[0];
		pc.color.g = color[1];
		pc.color.b = color[2];
		pc.color.a = 0.5f;

		// flag the marker to be added
		pc.action = Marker::ADD;
		pc.lifetime = ros::Duration(5.0); // seconds
		pc.points.clear();

		for (size_t j = 0; j < candidate.parts().size(); ++j)
		{
			ROS_INFO(
					"\tPart %zu: %d %d -> %d %d conf: %f", j, candidate.parts()[j].x, candidate.parts()[j].y, candidate.parts()[j].height, candidate.parts()[j].width, candidate.confidence()[j]);
			cv::Rect part = candidate.parts()[j];

			//clamp the part to the image
			part &= cv::Rect(cv::Point(0, 0), depth.size());
			cv::Point center(part.x + (part.width / 2),
					part.y + (part.height / 2));

			// Keep this independent from camera/point cloud resolution difference
			const Point3d point = camera_.projectPixelTo3dRay(center)
					* depth.at<float>(center);
//			const PointType& point = cloud.at(center.x, center.y);

			part_cloud_indices.push_back(center.x + center.y * cloud.width);
			geometry_msgs::Point out_point;
			out_point.x = point.x;
			out_point.y = point.y;
			out_point.z = point.z;
			pc.points.push_back(out_point);

//			mask(part) = GC_FGD;
//			mask(part - roi.tl()) = GC_PR_FGD;
//			mask.at<unsigned int>(cv::Point(part.x + part.width / 2, part.y + part.height / 2) - roi.tl()) = GC_FGD;
		}
		tl = camera_.projectPixelTo3dRay(Point2d(cube.tl().x, cube.tl().y))
				* cube.z;
		br = camera_.projectPixelTo3dRay(Point2d(cube.br().x, cube.br().y))
				* (cube.z + cube.depth);

		bounding_boxes.push_back(Rect3d(tl, br));
		scores.push_back(candidate.score());
		parts_centers.markers.push_back(pc);
	}

}

void PartsBasedDetectorNode::cleanCloud(const PointCloud::ConstPtr& cloud,
		const std::vector<Rect3d>& bounding_boxes, PointCloud& cleaned_cloud,
		PoseArray& object_poses)
{
	if (bounding_boxes.size() == 0)
	{
		return;
	}

	std::vector<pcl::PointIndices::Ptr> boxed_indices(bounding_boxes.size());
	boost::shared_ptr<std::vector<int> > final_indices(new std::vector<int>());

	cleaned_cloud.clear();
	object_poses.header = cloud->header;
	object_poses.poses.clear();

	// crop around each bounding box
	pcl::CropBox<PointType> crop_box;
	crop_box.setInputCloud(cloud);

	for (size_t i = 0; i < bounding_boxes.size(); ++i)
	{
		pcl::PointIndices::Ptr indices(new pcl::PointIndices());

		// expand a bit the box
		Rect3d bbox = bounding_boxes[i];
		bbox -= cv::Point3d(bbox.width * 0.1, bbox.height * 0.1,
				bbox.depth * 0.1);
		bbox.width *= 1.2;
		bbox.height *= 1.2;
		bbox.depth *= 1.2;

//		// reduce a bit the box
//		bbox += cv::Point3d(bbox.width * 0.1, bbox.height * 0.1, bbox.depth * 0.1);
//		bbox.width *= 0.8;
//		bbox.height *= 0.8;
//		bbox.depth *= 0.8;

		crop_box.setMin(
				Eigen::Vector4f(bbox.tl().x, bbox.tl().y, bbox.tl().z, 0));
		crop_box.setMax(
				Eigen::Vector4f(bbox.br().x, bbox.br().y, bbox.br().z, 0));

		crop_box.filter(indices->indices);

		// add the points belonging to the bbox
		boxed_indices[i] = indices;
	}

	ROS_INFO("OK2");

	// Extract different clusters to remove parts of other objects appearing into the bounding boxes
//	pcl::SeededHueSegmentation hue_clusterer;
//	hue_clusterer.setDeltaHue(0.75);
//	hue_clusterer.setClusterTolerance(0.025);
//	hue_clusterer.setInputCloud(cloud);

	pcl::EuclideanClusterExtraction<PointType> boxes_clusterer;
	boxes_clusterer.setInputCloud(cloud);
	boxes_clusterer.setClusterTolerance(0.010);

	ROS_INFO("OK3");

	for (int i = 0; i < bounding_boxes.size(); ++i)
	{
		pcl::PointIndices::Ptr hue_cluster(new pcl::PointIndices());
		std::vector<pcl::PointIndices> clusters;

		// first stage: clusters from the surroundings might remain
//		hue_clusterer.segment(*(boxed_indices[i]), *hue_cluster);
//		if(hue_cluster->indices.size() == 0)
//			continue;

		if (!boxed_indices[i] || boxed_indices[i]->indices.size() == 0)
		{
			ROS_INFO("Empty boxes indices %d", i);
			continue;
		}

		// second stage, keep the biggest one (should be the object)
		boxes_clusterer.setIndices(boxed_indices[i]);
//		boxes_clusterer.setIndices(hue_cluster);
		boxes_clusterer.extract(clusters);

		ROS_INFO("Extracted %zu clusters", clusters.size());

		// Keep only the maximum cluster
		if (clusters.size() > 0)
		{
			int max_idx = 0;
			for (int j = 1; j < clusters.size(); ++j)
			{
				if (clusters[j].indices.size()
						> clusters[max_idx].indices.size())
				{
					max_idx = j;
				}
			}

			//compute centroid
			Eigen::Vector4f centroid;
			Eigen::Matrix3f covMat;

			int point_count = 0;
			if ((point_count = pcl::computeMeanAndCovarianceMatrix(*cloud,
					clusters[max_idx].indices, covMat, centroid)) == 0)
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

			// add pose
			object_poses.poses.push_back(pose);

			// add cluster points to the final cloud
			final_indices->insert(final_indices->end(),
					clusters[max_idx].indices.begin(),
					clusters[max_idx].indices.end());
		}
	}

	// Assemble final cloud
	pcl::ExtractIndices<PointType> extract_indices;
	extract_indices.setInputCloud(cloud);
	extract_indices.setIndices(final_indices);

	extract_indices.filter(cleaned_cloud);
}
