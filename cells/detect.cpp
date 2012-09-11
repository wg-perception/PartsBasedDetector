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
#include <boost/scoped_ptr.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/ModelReader.h>
#include "PartsBasedDetector.hpp"
#include "FileStorageModel.hpp"
#include "Visualize.hpp"

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
    struct PartsBasedDetectorCell {

        // Parameters
        spore<bool> visualize_;
        spore<std::string> model_file_;

        // I/O
        spore<cv::Mat> color_, depth_, output_;
        spore<std::vector<PoseResult> > pose_results_;

        // the detector classes
        boost::scoped_ptr<Visualize> visualizer_;
        boost::scoped_ptr<PartsBasedDetector<float> > detector_;

        /*! @brief parameter callback
         *
         * @param db_documents the recognition database documents
         */
        void
        ParameterCallback(const object_recognition_core::db::Documents&) {}

        /*! @brief declare parameters used by the detector
         *
         * This method defines the mapping between the python members loaded
         * from the config file, and the members declared in this class. This
         * is called once at initialization, and again in instances of
         * dynamic reconfiguration
         *
         * @param params the parameters
         */
        static void
        declare_params(tendrils& params) {
            params.declare(&PartsBasedDetectorCell::visualize_, "visualize", "Visualize results", false);
            params.declare(&PartsBasedDetectorCell::model_file_, "model_file", "The path to the model file").required(true);

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
        static void
        declare_io(const tendrils&, tendrils& inputs, tendrils& outputs) {
            inputs.declare(&PartsBasedDetectorCell::color_, "image", "An rgb full frame image.");
            inputs.declare(&PartsBasedDetectorCell::depth_, "depth", "The 16bit depth image.");

            outputs.declare(&PartsBasedDetectorCell::pose_results_, "pose_results", "The results of object recognition");
            outputs.declare(&PartsBasedDetectorCell::output_, "image", "The results of object recognition");
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
        void
        configure(const tendrils&, const tendrils&, const tendrils&) {

            // create the model object and deserialize it
            FileStorageModel model;
            model.deserialize(*model_file_);

            // create the visualizer
            visualizer_.reset(new Visualize(model.name()));

            // create the PartsBasedDetector and distribute the model parameters
            detector_.reset(new PartsBasedDetector<float>);
            detector_->distributeModel(model);
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
        int
        process(const tendrils&, const tendrils&) {
            std::cout << "detector: process" << std::endl;

            std::vector<Candidate> candidates;
            detector_->detect(*color_, *depth_, candidates);

            if (true) {
            	if (candidates.size() > 0) {
            		Candidate::sort(candidates);
            		visualizer_->candidates(*color_, candidates, 1, *output_, true);
            	} else {
                    cvtColor(*color_, *output_, CV_RGB2BGR);
            	}
                cv::waitKey(30);
            }

            pose_results_->clear();
            return ecto::OK;
        }
    };
}

// register the ECTO cell
ECTO_CELL(object_recognition_by_parts_cells, parts_based_detector::PartsBasedDetectorCell, "Detector",
"Detection of objects by parts")
