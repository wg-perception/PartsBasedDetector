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
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/stereo_camera_model.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
#ifdef WITH_MATLABIO
	#include "MatlabIOModel.hpp"
#endif
#include "types.hpp"

using namespace cv;
using namespace std;

class PartsBasedDetectorNode {
private:
	// types
	typedef image_transport::SubscriberFilter ImageSubscriber;
	typedef image_transport::Publisher ImagePublisher;
	typedef message_filters::Subscriber<sensor_msgs::CameraInfo> CameraInfo;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> KinectSyncPolicy;
	typedef visualization_msgs::MarkerArray MarkerArray;
	typedef visualization_msgs::Marker Marker;
	typedef sensor_msgs::ImageConstPtr ImageConstPtr;
	typedef sensor_msgs::ImagePtr ImagePtr;
	typedef image_geometry::StereoCameraModel StereoCameraModel;
	typedef sensor_msgs::image_encodings::enc enc;

	// transports
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	StereoCameraModel camera_;
	message_filters::Synchronizer<KinectSyncPolicy> sync_;

	// subscribers
	ImageSubscriber image_sub_d_;       // the kinect subscriber
	ImageSubscriber image_sub_rgb_;     // the rgb camera subscriber

	// publishers
	ImagePublisher  image_pub_d_;       // the raw image publisher
	ImagePublisher  image_pub_rgb_;	 // the depth publisher
	ImagePublisher  mask_pub_;          // the object mask publisher
	ros::Publisher  bb_pub_;            // the bounding box publisher

	// PartsBasedDetector members
	PartsBasedDetector<float> pbd_;
	MarkerArray bb_markers_;



public:
	PartsBasedDetectorNode() :
							it_(nh_),
							image_sub_d_( it_, "camera/depth_registered/image_rect", 1),
							image_sub_rgb_( it_, "camera/rgb/image_rect_color", 1),
							info_sub_d_( it_, "camera/depth_registered/camera_info", 1),
							info_sub_rgb_( it_, "camera/rgb/camera_info", 1),
							sync_( KinectSyncPolicy(10), image_sub_d_, image_sub_rgb_) {

		// register the callback for synchronised depth and camera images
		sync.registerCallback( boost::bind(&PartsBasedDetectorNode::detectorCB, this, _1, _2 ) );

		// set the stereo camera parameters from the depth and rgb camera info
        camera_.fromCameraInfo(info_sub_rgb_, info_sub_d_);

        // initialise the publishers
        image_pub_d_ = it_.advertize("pbd/depth_rect", 1);
        image_pub_d_ = it_.advertize("pbd/candidates_rect_color", 1);
        mask_pub_    = it_.advertize("pbd/mask", 1);
        bb_pub_      = nh_.advertize("pbd/bounding_box", 1);
	}

    bool init(void);
    void clearMarkerArray(MarkerArray& markers, ros::Publisher& publisher);
    void messageBoundingBox(const vectorCandidate& candidates, cv::Mat& depth, const ImageConstPtr& msg, const StereoCameraModel& camera);
    void messageFrustum(const vectorCandidate& candidates);
    void messageImageRGB(const vectorCandidate& candidates, cv::Mat& rgb, const ImageConstPtr& msg_in);
    void messageImageDepth(cv::Mat& depth, const ImageConstPtr& msg_in);
    void messageMask(const vectorCandidate& candidates, cv::Mat& rgb, const ImageConstPtr& msg_in);
    void detectorCB(const ImageConstPtr &msg_d, const ImageConstPtr& msg_rgb);
};
