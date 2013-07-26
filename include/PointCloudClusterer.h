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
	typedef typename PointCloud::ConstPtr PointCloudConstPtr;
	typedef boost::function<cv::Point3d(cv::Point)> PointProjectFunc;

	/*! @brief this function computes the 3D bounding boxes around the candidates
	 *
	 * @param candidates the candidates returned by the parts based detector
	 * @param rgb the rgb image
	 * @param depth the depth image
	 * @param camera_model_projecter the function that projects the 2d pixel into a 3d ray
	 * @param cloud the pointcloud
	 * @param bounding_boxes the 3D bounding boxes for each candidate
	 * @param parts_centers a point cloud for each candidate, each point represents the center for a part
	 */
	static void computeBoundingBoxes(const std::vector<Candidate>& candidates,
			const cv::Mat& rgb, const cv::Mat& depth,
			PointProjectFunc camera_model_projecter,
			const typename PointCloud::ConstPtr cloud,
			std::vector<Rect3d>& bounding_boxes,
			std::vector<PointCloud>& parts_centers);

	/*! @brief this function uses the 3D bounding boxes to segment and extract a point cluster for each detected object
	 *
	 * @param cloud the input point cloud
	 * @param bounding_boxes the 3D bunding boxes
	 * @param object_clusters a vector of PointClouds is returned, a cluster for each bounding box
	 * @param object_centers the centroid of each cluster
	 */
	static void clusterObjects(const PointCloudConstPtr cloud,
			const std::vector<Rect3d>& bounding_boxes,
			std::vector<PointCloud>& object_clusters,
			std::vector<PointType>& object_centers);

	/*! @brief this function removes planes from the (organized) input cloud
	 *
	 * @param cloud the input point cloud
	 * @param cloud_no_plane the filtered input cloud (planes removed)
	 */
	static void organizedMultiplaneSegmentation(const PointCloudConstPtr& cloud,
			PointCloud& cloud_no_plane);
};

// To instantiate the template
#include "PointCloudClusterer.hpp"

#endif /* POINTCLOUDCLUSTERER_H_ */
