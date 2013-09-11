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
#include <string>
#include <iostream>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/stereo_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"

#ifdef WITH_MATLABIO
#include "MatlabIOModel.hpp"
#endif

#include "types.hpp"

namespace enc = sensor_msgs::image_encodings;

class PartsBasedDetectorNode {
private:
	// types
	typedef pcl::PointXYZRGB PointType;
	typedef pcl::PointCloud<PointType> PointCloud;
	typedef image_transport::SubscriberFilter ImageSubscriber;
	typedef image_transport::Publisher ImagePublisher;
	typedef sensor_msgs::CameraInfo CameraInfo;
	typedef sensor_msgs::CameraInfoConstPtr CameraInfoConstPtr;
	typedef message_filters::Subscriber<CameraInfo> CameraInfoSubscriber;
#if PCL_VERSION_COMPARE(<,1,7,0)
	typedef message_filters::Subscriber<PointCloud> PointCloudSubscriber;
#else
	typedef message_filters::Subscriber<sensor_msgs::PointCloud2> PointCloudSubscriber;
#endif
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
#if PCL_VERSION_COMPARE(<,1,7,0)
			sensor_msgs::Image, PointCloud> KinectSyncPolicy;
#else
			sensor_msgs::Image, sensor_msgs::PointCloud2> KinectSyncPolicy;
#endif
	typedef visualization_msgs::MarkerArray MarkerArray;
	typedef visualization_msgs::Marker Marker;
	typedef geometry_msgs::Pose Pose;
	typedef geometry_msgs::PoseArray PoseArray;
	typedef sensor_msgs::ImageConstPtr ImageConstPtr;
	typedef sensor_msgs::ImagePtr ImagePtr;
	typedef image_geometry::PinholeCameraModel PinholeCameraModel;
	typedef image_geometry::StereoCameraModel StereoCameraModel;

	// transports
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;

	// subscribers
	ImageSubscriber image_sub_d_;       // the kinect subscriber
	ImageSubscriber image_sub_rgb_;     // the rgb camera subscriber
	CameraInfoSubscriber info_sub_d_;	// the kinect info subscriber
	PointCloudSubscriber pointcloud_sub_;	// the pointcloud subscriber
	message_filters::Synchronizer<KinectSyncPolicy> sync_;

	// publishers
//	ImagePublisher image_pub_d_;       // the depth publisher
	ImagePublisher image_pub_rgb_;	 	// the raw image publisher
	ImagePublisher mask_pub_;          // the object mask publisher
//	ImagePublisher segmented_pub_;		// the segmented image publisher
	ros::Publisher bb_pub_;            // the bounding box publisher
	ros::Publisher cloud_pub_;			// the clustered cloud publisher
	ros::Publisher part_center_pub_;	// the parts center publisher
	ros::Publisher object_pose_pub_;	// the object poses publisher

	// PartsBasedDetector members
	PartsBasedDetector<double> pbd_;
	MarkerArray bounding_box_markers_;
	std::string ns_;
	std::string name_;
	bool remove_planes_;

	// camera parameters
	bool depth_camera_initialized_;
	CameraInfo depth_camera_;
	PinholeCameraModel camera_;

	//utility functions
	void hashStringToColor(const std::string& str, cv::Scalar& rgb);

public:
	PartsBasedDetectorNode() :
			nh_(),
			it_(nh_),
			image_sub_d_(it_, "image_depth_in", 1),
			image_sub_rgb_(it_, "image_rgb_in", 1),
			info_sub_d_(nh_, "depth_camera_info_in", 1),
			pointcloud_sub_(nh_, "cloud_in", 1),
			sync_(KinectSyncPolicy(50), image_sub_d_, image_sub_rgb_, pointcloud_sub_),
			ns_("/pbd/"),
			remove_planes_ (false),
			depth_camera_initialized_(false) {	}

	// initialisation
	bool init(void);

	// message construction
	void clearMarkerArray(MarkerArray& markers, ros::Publisher& publisher);
	void messageBoundingBox(const std::vector<Rect3d>& bounding_boxes,
			const ImageConstPtr& msg);
	void messageFrustum(const vectorCandidate& candidates);
	void messageImageRGB(const vectorCandidate& candidates, cv::Mat& rgb,
			const ImageConstPtr& msg_in);
	void messageImageDepth(cv::Mat& depth, const ImageConstPtr& msg_in);
	void messageMask(const vectorCandidate& candidates, cv::Mat& rgb,
			const ImageConstPtr& msg_in);
	void messageClusters(const std::vector<PointCloud>& clusters);
	void messagePoses(const std_msgs::Header& header, const std::vector<PointCloud>& parts_centers);

	// callbacks
	void depthCameraCallback(const CameraInfoConstPtr& info_msg);
	void detectorCallback(const ImageConstPtr &msg_d,
#if PCL_VERSION_COMPARE(<,1,7,0)
			const ImageConstPtr& msg_rgb, const PointCloud::ConstPtr& msg_cloud);
#else
			const ImageConstPtr& msg_rgb, const sensor_msgs::PointCloud2::ConstPtr& msg_cloud_in);
#endif
};
