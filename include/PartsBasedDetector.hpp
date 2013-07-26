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
 *  File:    PartsBasedDetector.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef PARTSBASEDDETECTOR_HPP_
#define PARTSBASEDDETECTOR_HPP_
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <boost/scoped_ptr.hpp>
#include "Parts.hpp"
#include "Model.hpp"
#include "Candidate.hpp"
#include "IFeatures.hpp"
#include "IConvolutionEngine.hpp"
#include "DynamicProgram.hpp"
#include "SearchSpacePruning.hpp"

/*! @mainpage PartsBasedDetector
 *
 * PartsBasedDetector is a visual object recognition technique described in the
 * following series of papers by Deva Ramanan:
 *
 * - X. Zhu and D. Ramanan. "Face detection, pose estimation and landmark localization
 * in the wild", Internation Conference on Computer Vision and Pattern Recognition
 * (CVPR), 2012
 *
 * - Y. Yang and D. Ramanan. "Articulated pose estimation using flexible mixtures of parts",
 * International Conference on Computer Vision and Pattern Recognition (CVPR), 2011
 *
 * - P. Felzenszwalb, R. Girshick, D. McAllester and D. Ramanan. "Object detection with
 * disciminatively trained part based models", Journal on Pattern Analysis and Machine
 * Intelligence (PAMI), 2010
 *
 * Holistic appearance of complex deformations are difficult to model directly, so parts
 * based models break a holistic detector into a series of smaller "part" detectors with
 * geometric constraints enforcing particular spatial relationships between the parts.
 * In this manner, deformation of individual parts can be assumed to be linear, and the
 * geometric constraints can be approximated by a linear subspace, spring-mass damper system,
 * etc.
 *
 * PartsBasedDetector can detect many types of objects including:
 * - Rigid objects (coffee mugs)
 * - Objects of common semantic class but different appearance (chairs)
 * - Deformable objects (faces)
 * - Articulated objects (human bodies)
 *
 * The model supports three main features:
 * - search across scale
 * - multiple components (different views of an object where a different subset of parts
 * might be visible in each view, frontal and profile faces for example)
 * - multiple mixtures (each part may have multiple components or "views" such as a closed
 * vs open hand, or a vertically oriented vs a horizontally oriented hand
 *
 * The capacity of the classifier is largely a product of the training data supplied
 *
 * Detecting objects using the PartsBasedDetector requires three main steps:
 * - Instantiation of model classes
 * - Detection
 * - Visualization
 *
 * The following example code shows a common way to perform this pipeline:
 * \code
 * // create the model object and deserialize it
 * MatlabIOModel model;
 * model.deserialize(argv[1]);
 *
 * // create the PartsBasedDetector and distribute the model parameters
 * PartsBasedDetector<double> pbd;
 * pbd.distributeModel(model);
 *
 * // load the image from file
 * Mat im = imread(argv[2]);
 *
 * // detect potential candidates in the image
 * vector<Candidate> candidates;
 * pbd.detect(im, candidates);
 *
 * // visualize the best 5 detection candidates
 * Visualize visualize(model.name());
 * if (candidates.size() > 0) {
 *  Mat canvas;
 * 	Candidate::sort(candidates);
 * 	visualize.candidates(im, candidates, 5, canvas, true);
 * 	visualize.image(canvas);
 * 	waitKey();
 * }
 * \endcode
 *
 * And that's all there is to it!
 *
 * Multiple pre-trained models are available. Currently included are:
 * - Human body detector
 * - Face detector
 *
 * Model training is currently only supported via Matlab code provided by
 * Deva Ramanan. MatlabIOModel provides a method for deserializing models
 * generated by Matlab and saved in the .Mat format.
 *
 * ----------
 *
 * This packaged is written and maintained by Hilton Bristow, Willow Garage
 * with the consent of Deva Ramanan. The package is released under a BSD
 * license. Please see the included license file for details and acknowledgement
 * of contributions.
 *
 *
 *
 * @class PartsBasedDetector
 * @brief The main object detection class
 * PartsBasedDetector is the main entry point for detecting objects. It has a single
 * method distributeModel() for setting up the detector parameters from a deserialized
 * model, and a method detect() for running the detection pipeline.
 *
 * @tparam T the detector precision. Should be one of float or double. On modern 64-bit
 * machines, the latter will likely be just as fast.
 */
template<typename T>
class PartsBasedDetector {
private:
	//! the name of the Part detector
	std::string name_;
	//! produces features, feature pyramids and compares features with Parts
	boost::scoped_ptr<IFeatures> features_;
	//! compares features with Parts
	boost::scoped_ptr<IConvolutionEngine> convolution_engine_;
	//! dynamic program to predict part positions and candidate likelihoods from raw scores
	DynamicProgram<T> dp_;
	//! the tree of Parts
	Parts parts_;
	//! the search space pruner
	SearchSpacePruning<T> ssp_;
public:
	PartsBasedDetector() {}
	virtual ~PartsBasedDetector() {}
	// public methods
	const std::string& name(void) const { return name_; }
	void detect(const cv::Mat& im, std::vector<Candidate>& candidates);
	void detect(const cv::Mat& im, const cv::Mat& depth, std::vector<Candidate>& candidates);
	void distributeModel(Model& model);
};

#endif /* PARTSBASEDDETECTOR_HPP_ */
