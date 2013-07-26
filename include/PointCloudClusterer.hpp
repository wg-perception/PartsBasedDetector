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

#ifndef POINTCLOUDCLUSTERER_HPP_
#define POINTCLOUDCLUSTERER_HPP_

#include "PointCloudClusterer.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <pcl/PointIndices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

template<typename PointType>
void PointCloudClusterer<PointType>::computeBoundingBoxes(
		const std::vector<Candidate>& candidates, const cv::Mat& rgb,
		const cv::Mat& depth, PointProjectFunc camera_model_projecter,
		const typename PointCloud::ConstPtr cloud,
		std::vector<Rect3d>& bounding_boxes,
		std::vector<PointCloud>& parts_centers)
{
	bounding_boxes.clear();
	parts_centers.clear();

	if (candidates.size() == 0)
	{
		return;
	}

	double ticks = (double) cv::getTickCount();

	bounding_boxes.resize(candidates.size(), Rect3d(0, 0, 0, 0, 0, 0));
	parts_centers.resize(candidates.size());

	for (size_t i = 0; i < candidates.size(); ++i)
	{
		const Candidate& candidate = candidates[i];

		const Rect3d cube = candidate.boundingBox3D(rgb, depth);
		cv::Point3d tl, br;

		if (isnan(cube.x) || isnan(cube.y) || isnan(cube.z) || isnan(cube.width)
				|| isnan(cube.height) || isnan(cube.depth))
		{
			std::cout << "Candidate " << i << " bounding box contains nans."
					<< std::endl;
			//ROS_INFO("Candidate %zu bounding box contains nans.", i);
			continue;
		}

		std::cout << "Score: " << candidate.score() << " - Cube data: "
				<< cube.tl().x << " " << cube.tl().y << " " << cube.z << " "
				<< cube.br().x << " " << cube.br().y << " " << cube.depth
				<< std::endl;
// 		ROS_INFO("Score: %f - Cube data: %f %f %f %f %f %f", candidate.score(),
// 				cube.tl().x, cube.tl().y, cube.z, cube.br().x, cube.br().y,
// 				cube.depth);

		for (size_t j = 0; j < candidate.parts().size(); ++j)
		{
// 			ROS_INFO("\tPart %zu: %d %d -> %d %d conf: %f", j,
// 					candidate.parts()[j].x, candidate.parts()[j].y,
// 					candidate.parts()[j].height, candidate.parts()[j].width,
// 					candidate.confidence()[j]);
			cv::Rect part = candidate.parts()[j];

			//clamp the part to the image
			part &= cv::Rect(cv::Point(0, 0), rgb.size());
			cv::Point center(part.x + (part.width / 2),
					part.y + (part.height / 2));

			double avg_depth = 0.0;
			for (int row_it = part.x; row_it < part.x + part.height; ++row_it)
			{
				const float *row_ptr = depth.ptr<float>(row_it);
				for (int col_it = part.y; col_it < part.y + part.width;
						++col_it)
				{
					avg_depth += row_ptr[col_it];
				}
			}

			if (part.width * part.height != 0)
			{
				avg_depth /= part.width * part.height;
			}

			// To keep this independent from camera/point cloud resolution difference:
			const cv::Point3d point = camera_model_projecter(center)
					* avg_depth; // depth.at<float>(center);
//			const PointType& out_point = cloud.at(center.x, center.y);

			PointType out_point;
			out_point.x = point.x;
			out_point.y = point.y;
			out_point.z = point.z;

			if (isnan(out_point.x) || isnan(out_point.y) || isnan(out_point.z))
			{
				parts_centers[i].is_dense = false;
			}

			parts_centers[i].push_back(out_point);
		}
		tl = camera_model_projecter(cv::Point2d(cube.tl().x, cube.tl().y))
				* cube.z;
		br = camera_model_projecter(cv::Point2d(cube.br().x, cube.br().y))
				* (cube.z + cube.depth);

		bounding_boxes[i] = Rect3d(tl, br);
	}

	std::cout << "Bounding boxes computation time: "
			<< ((double) cv::getTickCount() - ticks) / cv::getTickFrequency()
			<< std::endl;
}

template<typename PointType>
void PointCloudClusterer<PointType>::clusterObjects(
		const PointCloudConstPtr cloud,
		const std::vector<Rect3d>& bounding_boxes,
		std::vector<PointCloud>& object_clusters,
		std::vector<PointType>& object_centers)
{
	object_clusters.clear();
	object_centers.clear();

	if (bounding_boxes.size() == 0)
	{
		return;
	}

	double ticks = (double) cv::getTickCount();

	{
		PointType nan_point;
		nan_point.x = std::numeric_limits<float>::quiet_NaN();
		nan_point.y = std::numeric_limits<float>::quiet_NaN();
		nan_point.z = std::numeric_limits<float>::quiet_NaN();

		object_clusters.resize(bounding_boxes.size());
		object_centers.resize(bounding_boxes.size(), nan_point);

	}

	std::vector<pcl::PointIndices::Ptr> boxed_indices(bounding_boxes.size());
	//boost::shared_ptr<std::vector<int> > final_indices(new std::vector<int>());

	// crop around each bounding box
	pcl::CropBox<PointType> crop_box;
	crop_box.setInputCloud(cloud);

	for (size_t i = 0; i < bounding_boxes.size(); ++i)
	{
		pcl::PointIndices::Ptr indices(new pcl::PointIndices());

	    // expand a bit the box
		Rect3d bbox = bounding_boxes[i];

		if(bbox.volume() >= 1E-6) // 1 cubic centimeter
		{
			bbox -= cv::Point3d(bbox.width * 0.1, bbox.height * 0.1,
					bbox.depth * 0.1);
			bbox.width *= 1.2;
			bbox.height *= 1.2;
			bbox.depth *= 1.2;

			crop_box.setMin(
					Eigen::Vector4f(bbox.tl().x, bbox.tl().y, bbox.tl().z, 0));
			crop_box.setMax(
					Eigen::Vector4f(bbox.br().x, bbox.br().y, bbox.br().z, 0));

			crop_box.filter(indices->indices);
//			std::cout << "Cropped obj " << i << ", remaining indices: " << indices->indices.size() << std::endl;
		}
		else
		{
//			std::cout << "Bounding box for obj " << i << " empty." << std::endl;
		}
		// add the points belonging to the bbox
		boxed_indices[i] = indices;
	}

	std::cout << "Cropping ok..." << std::endl;

	// Extract different clusters to remove parts of other objects appearing into the bounding boxes
	pcl::EuclideanClusterExtraction<PointType> boxes_clusterer;
	boxes_clusterer.setInputCloud(cloud);
	boxes_clusterer.setClusterTolerance(0.010);

	pcl::ExtractIndices<PointType> extract_indices;
	extract_indices.setInputCloud(cloud);

	for (int i = 0; i < bounding_boxes.size(); ++i)
	{
		std::vector<pcl::PointIndices> clusters;

		if (!boxed_indices[i] || boxed_indices[i]->indices.size() == 0)
		{
// 			ROS_INFO("Empty boxes indices %d", i);
			continue;
		}

		std::cout << "Start clustering " << i << std::endl;
		// keep the biggest one (should be the object)
		boxes_clusterer.setIndices(boxed_indices[i]);
		boxes_clusterer.extract(clusters);
//		std::cout << "Clusterer extraction for object " << i << " succeeded with " << clusters.size() << " results" << std::endl;

		// ROS_INFO("Extracted %zu clusters", clusters.size());

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

			int point_count = 0;
			if ((point_count = pcl::compute3DCentroid(*cloud,
					clusters[max_idx].indices, centroid)) == 0)
			{
// 				ROS_WARN("Centroid not found...");
				continue;
			}

			PointType center_point;
			center_point.x = centroid.x();
			center_point.y = centroid.y();
			center_point.z = centroid.z();

			// add pose
			object_centers[i] = center_point;

			// extract cluster from main cloud
			extract_indices.setIndices(
					boost::make_shared<pcl::PointIndices>(clusters[max_idx]));
			extract_indices.filter(object_clusters[i]);
		}
	}

	std::cout << "Clustering time: "
			<< ((double) cv::getTickCount() - ticks) / cv::getTickFrequency()
			<< std::endl;
}

template<typename PointType>
void PointCloudClusterer<PointType>::organizedMultiplaneSegmentation(
		const PointCloudConstPtr& cloud, PointCloud& cloud_no_plane)
{
	pcl::PointCloud<pcl::Normal>::Ptr normals(
			new pcl::PointCloud<pcl::Normal>());
	pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.compute(*normals);

	pcl::OrganizedMultiPlaneSegmentation<PointType, pcl::Normal, pcl::Label> plane_segmentation;
	plane_segmentation.setInputCloud(cloud);
	plane_segmentation.setInputNormals(normals);
	plane_segmentation.setDistanceThreshold(0.02);
	//plane_segmentation.setAngularThreshold(pcl::deg2rad(5.0));
	plane_segmentation.setMaximumCurvature(0.001);
	plane_segmentation.setProjectPoints(true);

	std::vector<pcl::PlanarRegion<PointType>,
			Eigen::aligned_allocator<pcl::PlanarRegion<PointType> > > regions;
	std::vector<pcl::ModelCoefficients> model_coefficients;
	std::vector<pcl::PointIndices> inlier_indices;
	pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>());
	std::vector<pcl::PointIndices> label_indices;
	std::vector<pcl::PointIndices> boundary_indices;

	plane_segmentation.segmentAndRefine(regions, model_coefficients,
			inlier_indices, labels, label_indices, boundary_indices);

	boost::shared_ptr<std::vector<int> > inliers(new std::vector<int>());
	for (int i = 0; i < inlier_indices.size(); ++i)
	{
		inliers->insert(inliers->end(), inlier_indices[i].indices.begin(),
				inlier_indices[i].indices.end());
	}

	pcl::ExtractIndices<PointType> extract_indices;
	extract_indices.setInputCloud(cloud);
	extract_indices.setIndices(inliers);
	extract_indices.setNegative(true);
	extract_indices.filter(cloud_no_plane);
}

#endif /* POINTCLOUDCLUSTERER_HPP_ */
