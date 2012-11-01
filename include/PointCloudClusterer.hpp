/*
 * PointCloudClusterer.hpp
 *
 *  Created on: Oct 30, 2012
 *      Author: tcavallari
 */

#ifndef POINTCLOUDCLUSTERER_HPP_
#define POINTCLOUDCLUSTERER_HPP_

#include "PointCloudClusterer.h"
#include <vector>
#include <pcl/PointIndices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

template<typename PointType>
void PointCloudClusterer<PointType>::computeBoundingBoxes(
		const std::vector<Candidate>& candidates, const cv::Mat& rgb,
		const cv::Mat& depth, PointProjectFunc camera_model_projecter, const typename PointCloud::ConstPtr cloud,
		std::vector<Rect3d>& bounding_boxes,
		std::vector<PointCloud>& parts_centers)
{
	bounding_boxes.clear();
	parts_centers.clear();

	if (candidates.size() == 0)
	{
		return;
	}
	

	bounding_boxes.resize(candidates.size(), Rect3d(0, 0, 0, 0, 0, 0));
	parts_centers.resize(candidates.size());

	for (size_t i = 0; i < candidates.size(); ++i)
	{
		const Candidate& candidate = candidates[i];
		// TODO check
//		if (candidate.score() > 1.0)
//			continue;

		const Rect3d cube = candidate.boundingBox3D(rgb, depth);
		cv::Point3d tl, br;

		if (isnan(cube.x) || isnan(cube.y) || isnan(cube.z)
				|| isnan(cube.width) || isnan(cube.height)
				|| isnan(cube.depth))
		{
			//ROS_INFO("Candidate %zu bounding box contains nans.", i);
			continue;
		}

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

			// TODO Keep this independent from camera/point cloud resolution difference
			const cv::Point3d point = camera_model_projecter(center) * depth.at<float>(center);
//			const PointType& out_point = cloud.at(center.x, center.y);

			PointType out_point;
			out_point.x = point.x;
			out_point.y = point.y;
			out_point.z = point.z;

			parts_centers[i].push_back(out_point);
		}
		tl = camera_model_projecter(cv::Point2d(cube.tl().x, cube.tl().y)) * cube.z;
		br = camera_model_projecter(cv::Point2d(cube.br().x, cube.br().y)) * (cube.z + cube.depth);

		bounding_boxes[i] = Rect3d(tl, br);
	}
}

template<typename PointType>
void PointCloudClusterer<PointType>::clusterObjects(
		const typename PointCloud::ConstPtr cloud,
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

	// Extract different clusters to remove parts of other objects appearing into the bounding boxes
//	pcl::SeededHueSegmentation hue_clusterer;
//	hue_clusterer.setDeltaHue(0.75);
//	hue_clusterer.setClusterTolerance(0.025);
//	hue_clusterer.setInputCloud(cloud);

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

// keep the biggest one (should be the object)
		boxes_clusterer.setIndices(boxed_indices[i]);
		boxes_clusterer.extract(clusters);

// 		ROS_INFO("Extracted %zu clusters", clusters.size());

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
// 				ROS_WARN("Centroid not found...");
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

			PointType center_point;
			center_point.x = centroid.x();
			center_point.y = centroid.y();
			center_point.z = centroid.z();

			// add pose
			object_centers[i] = center_point;

			// extract cluster from main cloud
			extract_indices.setIndices(boost::make_shared<pcl::PointIndices>(clusters[max_idx]));
			extract_indices.filter(object_clusters[i]);
		}
	}
}

#endif /* POINTCLOUDCLUSTERER_HPP_ */
