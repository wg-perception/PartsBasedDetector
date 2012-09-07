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

 namespace enc = sensor_msgs::image_encodings;
 using namespace cv;
 using namespace std;

 class PartsBasedDetectorNode {
 private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    typedef image_transport::SubscriberFilter ImageSubscriber;
    typedef message_filters::Subscriber<sensor_msgs::CameraInfo> CameraInfo;
    image_geometry::StereoCameraModel cam_model_;
    ImageSubscriber image_sub_d_;       // the kinect subscriber
    ImageSubscriber image_sub_rgb_;     // the rgb camera subscriber
    ros::Publisher  cloud_pub_;         // the detected object point cloud publiser
    CameraInfo      info_sub_d_;        // the kinect info subscriber
    CameraInfo      info_sub_rgb_;      // the rgb camera info subscriber

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> KinectSyncPolicy;
    message_filters::Synchronizer<KinectSyncPolicy> sync_;

private:
    PartsBasedDetectorNode() :  it_(nh_),
                                image_sub_d_( it_, "camera/depth_registered/image_rect", 1),
                                image_sub_rgb_( it_, "camera/rgb/image_rect_color", 1),
                                cloud_pub_ ( nh_.advertise<sensor_msgs::PointCloud2> "pbd/cloud", 1),
                                info_sub_d_( it_, "camera/depth_registered/camera_info", 1),
                                info_sub_rgb_( it_, "camera/rgb/camera_info", 1),
                                sync_( KinectSyncPolicy(10), image_sub_d_, image_sub_rgb_) {
        // register the callback for synchronised depth and camera images
        sync.registerCallback( boost::bind(&PartsBasedDetectorNode::detectorCB, this, _1, _2 ) );
        // set the stereo camera parameters from the depth and rgb camera info
        cam_model_.fromCameraInfo(info_sub_rgb_, info_sub_d_);
    }

    
    void detectorCB(const sensor_msgs::ImageConstPtr &msg_d, const sensor_msgs::ImageConstPtr &msg_rgb);
