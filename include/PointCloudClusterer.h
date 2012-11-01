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
 * PointCloudClusterer.hpp
 *
 *  Created on: Oct 30, 2012
 *      Author: tcavallari
 */

#ifndef POINTCLOUDCLUSTERER_H_
#define POINTCLOUDCLUSTERER_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "Candidate.hpp"
#include "Rect3.hpp"

template<typename PointType>
class PointCloudClusterer
{

public:
	typedef pcl::PointCloud<PointType> PointCloud;
  typedef boost::function<cv::Point3d (cv::Point)> PointProjectFunc;

	static void computeBoundingBoxes(const std::vector<Candidate>& candidates,
			const cv::Mat& rgb, const cv::Mat& depth, PointProjectFunc camera_model_projecter, const typename PointCloud::ConstPtr cloud,
			std::vector<Rect3d>& bounding_boxes, std::vector<PointCloud>& parts_centers);

	static void clusterObjects(const typename PointCloud::ConstPtr cloud,
			const std::vector<Rect3d>& bounding_boxes,
			std::vector<PointCloud>& object_clusters, std::vector<PointType>& object_centers);
};

// To instantiate the template
#include "PointCloudClusterer.hpp"

#endif /* POINTCLOUDCLUSTERER_H_ */
