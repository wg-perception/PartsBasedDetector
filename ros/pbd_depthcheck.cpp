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

#include <ros/ros.h>
#include <image_geometry/stereo_camera_model.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace cv;
using namespace std;
using namespace image_geometry;

class PartsBasedDetector {
private:
    double depthUncertainty(double Z, const image_geometry::StereoCameraModel &cam_model);  
public:
    void filterResponseByDepth(const Mat &src, Mat &dst, const Mat& depth, const double u, const double X, const StereoCameraModel &cam_model);
};

double PartsBasedDetector::depthUncertainty(double Z, const image_geometry::StereoCameraModel &cam_model) {
    double disparity = cam_model.getDisparity(Z);
    return cam_model.getZ(disparity) - cam_model.getZ(disparity+1.0f/8.0f);
}

void PartsBasedDetector::filterResponseByDepth(const Mat &src, Mat &dst, const Mat& depth, const double u, const double X, const StereoCameraModel &cam_model) {

    // allocate the output array if necessary
    dst.create(src.size(), src.type());
    src.copyTo(dst);
    cv::Mat_<float> depthf = depth;
    cv::Mat_<float> dstf = dst;

    // create a lookup table to transform source pixels to depth pixels
    Mat_<int> luti(src.rows, 1, DataType<int>::type);
    Mat_<int> lutj(src.cols, 1, DataType<int>::type);
    float ifactor = depth.rows / src.rows;
    float jfactor = depth.cols / src.cols;
    for (int i = 0; i < src.rows; ++i) luti(i, 0) = (float)i*ifactor;
    for (int j = 0; j < src.cols; ++j) lutj(j, 0) = (float)j*jfactor;

    // calculate the depth of the part in real-world coordinates,
    // given the 3d width of the part, the focal length of the camera and the part width in the image
    double us = u*jfactor;
    double Z  = cam_model.left().fx()*X/us;
    double uncertainty = depthUncertainty(Z, cam_model);

    double Z_min = Z - 10*uncertainty;
    double Z_max = Z + 10*uncertainty;

    // for each pixel, set the depth to being valid or invalid
    #ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    #endif
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            float dij = depthf(luti(i, 0), lutj(j, 0));
            if ((dij > Z_max && dij < StereoCameraModel::MISSING_Z) || (dij < Z_min)) dstf(i, j) = -2;
        }
    }
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "Depth Check");
    PartsBasedDetector pbd;

    // create a camera model
    image_geometry::StereoCameraModel cam_model;
    cam_model.fromCameraInfo(c1_info, c2_info);

    Mat feature = Mat::ones(100, 100, DataType<float>::type);
    Mat feature2;
    Mat depth = Mat::zeros(640, 480, DataType<float>::type);
    depth(Rect(0, 0, 320, 480)) = Mat::ones(320, 480, DataType<float>::type);
    pbd.filterResponseByDepth(feature, feature2, depth, 4.0f, 0.1f, cam_model);

    imshow("Original features");
    imshow("Depth consistent features");
    ros::spin();
}
