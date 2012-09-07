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
#include "node.hpp"
using namespace cv;
using namespace std;

 void convertMatToPointCloud(const Mat &points_mat, PointCloud2Ptr &points_msg, const Header header) {
    
    // populate the point cloud
    Mat_<Vec3f> mat = points_mat;
    points_msg = boost::make_shared<PointCloud2>();
    points_msg->header = header;
    points_msg->height = mat.rows;
    points_msg->width = mat.cols;
    points_msg->fields.resize (3);
    points_msg->fields[0].name = "x";
    points_msg->fields[0].offset = 0;
    points_msg->fields[0].count = 1;
    points_msg->fields[0].datatype = PointField::FLOAT32;
    points_msg->fields[1].name = "y";
    points_msg->fields[1].offset = 4;
    points_msg->fields[1].count = 1;
    points_msg->fields[1].datatype = PointField::FLOAT32;
    points_msg->fields[2].name = "z";
    points_msg->fields[2].offset = 8;
    points_msg->fields[2].count = 1;
    points_msg->fields[2].datatype = PointField::FLOAT32;
    //points_msg->is_bigendian = false; ???
    static const int STEP = 16;
    points_msg->point_step = STEP;
    points_msg->row_step = points_msg->point_step * points_msg->width;
    points_msg->data.resize (points_msg->row_step * points_msg->height);
    points_msg->is_dense = false; // there may be invalid points

    // copy the data across
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    int offset = 0;
    for (int v = 0; v < mat.rows; ++v) {
        for (int u = 0; u < mat.cols; ++u, offset += STEP) {
            if (isValidPoint(mat(v,u))) {
                // x,y,z,rgba
                memcpy (&points_msg->data[offset + 0], &mat(v,u)[0], sizeof (float));
                memcpy (&points_msg->data[offset + 4], &mat(v,u)[1], sizeof (float));
                memcpy (&points_msg->data[offset + 8], &mat(v,u)[2], sizeof (float));
            } else {
                memcpy (&points_msg->data[offset + 0], &bad_point, sizeof (float));
                memcpy (&points_msg->data[offset + 4], &bad_point, sizeof (float));
                memcpy (&points_msg->data[offset + 8], &bad_point, sizeof (float));
            }
        }
    }
 }



 void PartsBasedDetectorNode::detectorCB(const sensor_msgs::ImageConstPtr &msg_d, const sensor_msgs::ImageConstPtr &msg_rgb) {

    // update the stereo camera parameters from the depth and rgb camera info
    cam_model_.fromCameraInfo(info_sub_rgb_, info_sub_d_);

    // convert the ROS image payloads to OpenCV structures
    cv_bridge::CvImagePtr cv_ptr_d;
    cv_bridge::CvImagePtr cv_ptr_rgb;
    try {
        cv_ptr_d   = cv_bridge::toCvCopy(msg_d, enc::TYPE_32FC1);
        cv_ptr_rgb = cv_bridge::toCvCopy(msg_rgb, enc::TYPE_8UC3);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s\n", e.what());
        return;
    }

    // strip out the matrices
    Mat image_d   = cv_ptr_d->image;
    Mat image_rgb = cv_ptr_rgb->image;

    // run the underlying OpenCV detection algorithm
    // TODO: write the underlying OpenCV detection algorithm :p
    vector<Rect> bounds;
    int ndetections = bounds.size();

    // produce a PointCloud2 if there are any detections
    if (ndetections > 0) {
        Mat cloud;
        cam_model_.projectDisparityTo3d(image_d, cloud);
        PointCloud2Ptr points_msg;
        convertMat2PointCloud(cloud, points_msg);
        cloud_pub_.publish(points_msg);
    }
    


 }
