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
 *
 *  File:    detect.cpp
 *  Author:  Hilton Bristow
 *  Created: Aug 28, 2012
 */
#include <string>
#include <boost/scoped_ptr.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>

#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/ModelReader.h>
#include <ecto_pcl/ecto_pcl.hpp>
#include <ecto_pcl/pcl_cell.hpp>
#include <pcl/point_cloud.h>
#include <ecto_image_pipeline/pinhole_camera_model.h>

#include "PartsBasedDetector.hpp"
#include "FileStorageModel.hpp"
#include "Visualize.hpp"
#include "Rect3.hpp"
#include "PointCloudClusterer.h"

#if PCL_VERSION_COMPARE(>=,1,7,0)
#include <pcl_conversions/pcl_conversions.h>
#endif

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectDbPtr;

namespace parts_based_detector
{

/*! @class PartsBasedDetectorCell
 *  @brief ECTO cell to wrap the PartsBasedDetector
 *
 *  This class implements the detection cell of the ECTO
 *  pipeline using the PartsBasedDetector method
 */
struct PartsBasedDetectorCell: public object_recognition_core::db::bases::ModelReaderBase
{
	typedef pcl::PointXYZ PointType;
	typedef pcl::PointCloud<PointType> PointCloud;

	// Parameters
	spore<bool> visualize_;
	spore<bool> remove_planes_;
	spore<std::string> model_file_;
	spore<float> max_overlap_;
	spore<ObjectDbPtr> object_db_;

	// I/O
	spore<PointCloud::ConstPtr> input_cloud_;
	spore<cv::Mat> color_, depth_, camera_intrinsics_, output_;
	spore<std::vector<PoseResult> > pose_results_;

	// the detector classes
	boost::scoped_ptr<Visualize> visualizer_;
	boost::scoped_ptr<PartsBasedDetector<double> > detector_;

	// model_name
	ObjectId model_name_;

	/*! @brief parameter callback
	 *
	 * @param db_documents the recognition database documents
	 */
	void parameter_callback(const object_recognition_core::db::Documents&)
	{
	}

	/*! @brief declare parameters used by the detector
	 *
	 * This method defines the mapping between the python members loaded
	 * from the config file, and the members declared in this class. This
	 * is called once at initialization, and again in instances of
	 * dynamic reconfiguration
	 *
	 * @param params the parameters
	 */
	static void declare_params(tendrils& params)
	{
		object_recognition_core::db::bases::declare_params_impl(params, "PartsBased");
		params.declare(&PartsBasedDetectorCell::visualize_, "visualize",
				"Visualize results", false);
		params.declare(&PartsBasedDetectorCell::remove_planes_, "remove_planes",
				"The cell should remove planes from the scene before the cluster extraction", false);
		params.declare(&PartsBasedDetectorCell::model_file_, "model_file",
				"The path to the model file").required(true);
		params.declare(&PartsBasedDetectorCell::max_overlap_, "max_overlap",
				"The maximum overlap allowed between object detections", 0.1);
	}

	/*! @brief declare the I/O of the detector
	 *
	 * Declares the input/output requirements of the detector so that calls
	 * to process() are made when all input dependencies are met, and referrals
	 * of the outputs are called once the process() method returns
	 *
	 * @param params the parameters
	 * @param inputs a hook to the inputs
	 * @param outputs a hook to the outputs
	 */
	static void declare_io(const tendrils& params, tendrils& inputs,
			tendrils& outputs)
	{
		inputs.declare(&PartsBasedDetectorCell::color_, "image",
				"An rgb full frame image.");
		inputs.declare(&PartsBasedDetectorCell::depth_, "depth",
				"The 16bit depth image.");
		inputs.declare(&PartsBasedDetectorCell::camera_intrinsics_, "K",
				"The camera intrinsics matrix.");
		inputs.declare(&PartsBasedDetectorCell::input_cloud_, "input_cloud",
				"The input point cloud.");

		outputs.declare(&PartsBasedDetectorCell::pose_results_, "pose_results",
				"The results of object recognition");
		outputs.declare(&PartsBasedDetectorCell::output_, "image",
				"The results of object recognition");
	}

	/*! @brief configure the detector state
	 *
	 * The configure method is called once when the ECTO cell is launched,
	 * and is designed to initialise the detector state and load models,
	 * parameters, etc.
	 *
	 * @param params the parameters made available through the config file and
	 * python bindings
	 * @param inputs for initializing inputs, if necessary
	 * @param outputs for initializing outputs, if necessary
	 */
	void configure(const tendrils& params, const tendrils& inputs,
			const tendrils& outputs)
	{
		configure_impl();
		std::cout << "MODEL: " << *model_file_ << std::endl;
		// create the model object and deserialize it
		FileStorageModel model;
		model.deserialize(*model_file_);

		// create the visualizer
		visualizer_.reset(new Visualize(model.name()));

		// create the PartsBasedDetector and distribute the model parameters
		detector_.reset(new PartsBasedDetector<double>);
		detector_->distributeModel(model);

		// set the model_name
		model_name_ = model.name();
	}

	/*! @brief project a pixel from the 2D image into a 3D ray
	 *
	 * @param camera the pinhole camera model used for the projection
	 * @param pixel the pixel to project
	 * @return the ray that intersects the pixel
	 */
	cv::Point3d projectPixelToRay(image_pipeline::PinholeCameraModel camera,
			cv::Point2d pixel)
	{
		Eigen::Vector3d point = camera.projectPixelTo3dRay(
				Eigen::Vector2d(pixel.x, pixel.y));

		return cv::Point3d(point.x(), point.y(), point.z());
	}

	/*! @brief the main processing callback of the ECTO pipeline
	 *
	 * this method is called once all input dependencies are satisfied.
	 * The PartsBasedDetector has two input dependencies: a color image and depth image,
	 * both retrieved from the Kinect. If any detection candidates are found,
	 * the bounding boxes, detection confidences and object ids are returned
	 *
	 * @param inputs the input tendrils
	 * @param outputs the output tendrils
	 * @return
	 */
	int process(const tendrils& inputs, const tendrils& outputs)
	{
		std::cout << "detector: process" << std::endl;

		pose_results_->clear();

		image_pipeline::PinholeCameraModel camera_model;
		camera_model.setParams(color_->size(), *camera_intrinsics_, cv::Mat(),
				cv::Mat(), cv::Mat());

		std::vector<Candidate> candidates;
		detector_->detect(*color_, *depth_, candidates);

		if (candidates.size() == 0)
		{
			if (*visualize_)
			{
				cv::cvtColor(*color_, *output_, CV_RGB2BGR);
				//cv::waitKey(30);
			}

			return ecto::OK;
		}

		Candidate::sort(candidates);
		Candidate::nonMaximaSuppression(*color_, candidates, *max_overlap_);

		if (*visualize_)
		{
			visualizer_->candidates(*color_, candidates, candidates.size(), *output_, true);
			cv::cvtColor(*output_, *output_, CV_RGB2BGR);
			//cv::waitKey(30);
		}

		std::vector<Rect3d> bounding_boxes;
		std::vector<PointCloud> parts_centers;

		typename PointCloudClusterer<PointType>::PointProjectFunc projecter =
				boost::bind(&PartsBasedDetectorCell::projectPixelToRay, this,
						camera_model, _1);
		PointCloudClusterer<PointType>::computeBoundingBoxes(candidates,
				*color_, *depth_, projecter, *input_cloud_, bounding_boxes,
				parts_centers);


		// output clusters
		std::vector<PointType> object_centers;
		std::vector<PointCloud> clusters;

		// remove planes from input cloud if needed
		if(*remove_planes_)
		{
			PointCloud::Ptr clusterer_cloud (new PointCloud());
			PointCloudClusterer<PointType>::organizedMultiplaneSegmentation(*input_cloud_, *clusterer_cloud);
			PointCloudClusterer<PointType>::clusterObjects(clusterer_cloud,
					bounding_boxes, clusters, object_centers);
		}
		else
		{
			PointCloudClusterer<PointType>::clusterObjects(*input_cloud_,
					bounding_boxes, clusters, object_centers);
		}




		// compute poses (centroid of part centers)

		// for each object
		for (int object_it = 0; object_it < candidates.size(); ++object_it)
		{
			if(std::isnan(object_centers[object_it].x) || std::isnan(object_centers[object_it].y) || std::isnan(object_centers[object_it].z))
				continue;

			PoseResult result;

			// no db for now, only one model
			result.set_object_id(*object_db_, model_name_);
			result.set_confidence(candidates[object_it].score());

			// set the clustered cloud's center as a center...
			result.set_T(Eigen::Vector3f(object_centers[object_it].getVector3fMap()));

//			// For the rotation a minimum of two parts is needed
//			if (part_centers_cloud.size() >= 2 &&
//					!pcl_isnan(part_centers_cloud[0].x) &&
//					!pcl_isnan(part_centers_cloud[0].y) &&
//					!pcl_isnan(part_centers_cloud[0].z) &&
//					!pcl_isnan(part_centers_cloud[1].x) &&
//					!pcl_isnan(part_centers_cloud[1].y) &&
//					!pcl_isnan(part_centers_cloud[1].z))
//			{
//				Eigen::Vector3f center(centroid.block<3, 1>(0, 0));
//
//				Eigen::Vector3f x_axis(
//						part_centers_cloud[0].getVector3fMap() - center);
//				x_axis.normalize();
//				Eigen::Vector3f z_axis =
//						(x_axis.cross(
//								part_centers_cloud[1].getVector3fMap() - center)).normalized();
//
//				Eigen::Vector3f y_axis = x_axis.cross(z_axis); // should be normalized
//
//				Eigen::Matrix3f rot_matr;
//				rot_matr << z_axis, y_axis, -x_axis;
//				//rot_matr.transposeInPlace();
//
//				result.set_R(rot_matr);
//			}
//			else
			{
				result.set_R(Eigen::Quaternionf(1, 0, 0, 0));
			}

			// Only one point of view for this object...
			sensor_msgs::PointCloud2Ptr cluster_cloud (new sensor_msgs::PointCloud2());
	        std::vector<sensor_msgs::PointCloud2ConstPtr> ros_clouds (1);

#if PCL_VERSION_COMPARE(<,1,7,0)
	        pcl::toROSMsg(clusters[object_it], *(cluster_cloud));
#else
        ::pcl::PCLPointCloud2 pcd_tmp;
        ::pcl::toPCLPointCloud2(clusters[object_it], pcd_tmp);
        pcl_conversions::fromPCL(pcd_tmp, *(cluster_cloud));
#endif
	        ros_clouds[0] = cluster_cloud;
	        result.set_clouds(ros_clouds);

			std::vector<PointCloud, Eigen::aligned_allocator<PointCloud> > object_cluster (1);
			object_cluster[0] = clusters[object_it];

			pose_results_->push_back(result);
		}

		return ecto::OK;
	}
};
}

// register the ECTO cell
ECTO_CELL(object_recognition_by_parts_cells,
		parts_based_detector::PartsBasedDetectorCell, "Detector",
		"Detection of objects by parts")
