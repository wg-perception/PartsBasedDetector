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

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectDb;

namespace parts_based_detector {

/*! @class PartsBasedDetectorCell
 *  @brief ECTO cell to wrap the PartsBasedDetector
 *
 *  This class implements the detection cell of the ECTO
 *  pipeline using the PartsBasedDetector method
 */
struct PartsBasedDetectorCell: public object_recognition_core::db::bases::ModelReaderImpl {

	// Parameters
	spore<bool> visualize_;
	spore<std::string> model_file_;
	spore<float> max_overlap_;

	// I/O
	//spore<ecto::pcl::PointCloud> input_cloud_;
	spore<cv::Mat> color_, depth_, camera_intrinsics_, output_;
	spore<std::vector<PoseResult> > pose_results_;
	spore<std::vector<ecto::pcl::PointCloud> > object_clusters_;

	// the detector classes
	boost::scoped_ptr<Visualize> visualizer_;
	boost::scoped_ptr<PartsBasedDetector<double> > detector_;
  
  // model_name
  ObjectId model_name_;

	/*! @brief parameter callback
	 *
	 * @param db_documents the recognition database documents
	 */
	void ParameterCallback(const object_recognition_core::db::Documents&) {
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
	static void declare_params(tendrils& params) {
		params.declare(&PartsBasedDetectorCell::visualize_, "visualize",
				"Visualize results", false);
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
			tendrils& outputs) {
		inputs.declare(&PartsBasedDetectorCell::color_, "image",
				"An rgb full frame image.");
		inputs.declare(&PartsBasedDetectorCell::depth_, "depth",
				"The 16bit depth image.");
    inputs.declare(&PartsBasedDetectorCell::camera_intrinsics_, "K", "The camera intrinsics matrix.");
    //already declared by PclCell
		//inputs.declare(&PartsBasedDetectorCell::input_cloud_, "input",
		//		"The input point cloud.");

		outputs.declare(&PartsBasedDetectorCell::pose_results_, "pose_results",
				"The results of object recognition");
		outputs.declare(&PartsBasedDetectorCell::output_, "image",
				"The results of object recognition");
		outputs.declare(&PartsBasedDetectorCell::object_clusters_, "clusters",
				"A vector containing a PointCloud for each recognized object");
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
	void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs) {

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
	
	cv::Point3d projectPixelToRay(image_pipeline::PinholeCameraModel camera, cv::Point2d pixel)
  {
    Eigen::Vector3d point = camera.projectPixelTo3dRay(Eigen::Vector2d(pixel.x, pixel.y));
    
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
  template<typename PointType>
  int process(const tendrils& inputs, const tendrils& outputs, const boost::shared_ptr<const pcl::PointCloud<PointType> >& input_cloud) 
  {
    // for comfort
    typedef pcl::PointCloud<PointType> PointCloud;
    //typedef PointCloudClusterer<PointType> PointCloudClusterer;
    
		std::cout << "detector: process" << std::endl;
    
    pose_results_->clear();
    object_clusters_->clear();
    
    image_pipeline::PinholeCameraModel camera_model;
    camera_model.setParams(color_->size(), *camera_intrinsics_, cv::Mat(), cv::Mat(), cv::Mat());

		std::vector<Candidate> candidates;
		detector_->detect(*color_, *depth_, candidates);

		if (candidates.size() == 0) {
			if (*visualize_) {
				cv::cvtColor(*color_, *output_, CV_RGB2BGR);
				cv::waitKey(30);
			}

			return ecto::OK;
		}

		Candidate::sort(candidates);
		Candidate::nonMaximaSuppression(*color_, candidates, *max_overlap_);

		if (*visualize_) {
			visualizer_->candidates(*color_, candidates, 1, *output_, true);
			cv::waitKey(30);
		}
    
    std::vector<Rect3d> bounding_boxes;
    std::vector<PointCloud> parts_centers;
    
    typename PointCloudClusterer<PointType>::PointProjectFunc projecter = boost::bind(&PartsBasedDetectorCell::projectPixelToRay, this, camera_model, _1);
    PointCloudClusterer<PointType>::computeBoundingBoxes(candidates, *color_, *depth_, projecter, input_cloud, bounding_boxes, parts_centers);
    
    // output clusters (forget about object_centers because it's computed using the cluster cloud)
    std::vector<PointType> object_centers;
    std::vector<PointCloud> clusters;
    PointCloudClusterer<PointType>::clusterObjects(input_cloud, bounding_boxes, clusters, object_centers);
    
    // compute poses (centroid of part centers)    
    
    // for each object
    for (int object_it = 0; object_it < candidates.size(); ++object_it)
    {
      PoseResult result;
      
      // no db for now, only one model
      result.set_object_id(ObjectDb(), model_name_);
      result.set_confidence(candidates[object_it].score());      
      
      // current cloud
      const PointCloud& part_centers_cloud = parts_centers[object_it];
      
      //compute centroid
      Eigen::Vector4f centroid;
           
      int point_count = 0;
      if ((point_count = pcl::compute3DCentroid(part_centers_cloud, centroid)) == 0)
      {
        //ROS_WARN("Centroid not found...");
        continue;
      }
      
      result.set_T(centroid);
      
      pose_results_->push_back(result);
      object_clusters_->push_back(ecto::pcl::xyz_cloud_variant_t(clusters[object_it].makeShared()));
    }  
    
		return ecto::OK;
	}
};
}

// register the ECTO cell
ECTO_CELL(object_recognition_by_parts_cells, ecto::pcl::PclCell<parts_based_detector::PartsBasedDetectorCell>, "Detector",
		"Detection of objects by parts")
