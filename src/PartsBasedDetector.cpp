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
 *  File:    PartsBasedDetector.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#include "PartsBasedDetector.hpp"
#include "nms.hpp"
#include "HOGFeatures.hpp"
#include "SpatialConvolutionEngine.hpp"
using namespace cv;
using namespace std;

/*! @brief search an image for potential candidates
 *
 * calls detect(const Mat& im, const Mat&depth=Mat(), vector<Candidate>& candidates);
 *
 * @param im the input color or grayscale image
 * @param candidates the output vector of detection candidates above the threshold
 */
template<typename T>
void PartsBasedDetector<T>::detect(const cv::Mat& im, vectorCandidate& candidates) {
	detect(im, Mat(), candidates);
}

/*! @brief search an image for potential object candidates
 *
 * This is the main entry point to the detection pipeline. Given an instantiated an populated model,
 * this method takes an input image, and attempts to find all instances of an object in that image.
 * The object, number of scales, detection confidence, etc are all defined through the Model.
 *
 * @param im the input color or grayscale image
 * @param depth the image depth image, used for depth consistency and search space pruning
 * @param candidates the output vector of detection candidates above the threshold
 */
template<typename T>
void PartsBasedDetector<T>::detect(const Mat& im, const Mat& depth, vectorCandidate& candidates) {

	// calculate a feature pyramid for the new image
	vectorMat pyramid;
	features_->pyramid(im, pyramid);

	// convolve the feature pyramid with the Part experts
	// to get probability density for each Part
	vector2DMat pdf;
	convolution_engine_->pdf(pyramid, pdf);

	// use dynamic programming to predict the best detection candidates from the part responses
	vector4DMat Ix, Iy, Ik;
	vector2DMat rootv, rooti;
	dp_.min(parts_, pdf, Ix, Iy, Ik, rootv, rooti);

	// suppress non-maximal candidates
	//ssp_.nonMaxSuppression(rootv, features_->scales());

	// walk back down the tree to find the part locations
	dp_.argmin(parts_, rootv, rooti, features_->scales(), Ix, Iy, Ik, candidates);

	if (!depth.empty()) {
		//ssp_.filterCandidatesByDepth(parts_, candidates, depth, 0.03);
	}

}

/*! @brief Distribute the model parameters to the PartsBasedDetector classes
 *
 * @param model the monolithic model containing the deserialization of all model parameters
 */
template<typename T>
void PartsBasedDetector<T>::distributeModel(Model& model) {

	// the name of the Part detector
	name_ = model.name();

	// initialize the Feature engine
	features_.reset(new HOGFeatures<T>(model.binsize(), model.nscales(), model.flen(), model.norient()));

	//initialise the convolution engine
	convolution_engine_.reset(new SpatialConvolutionEngine(DataType<T>::type, model.flen()));

	// make sure the filters are of the correct precision for the Feature engine
	const size_t nfilters = model.filters().size();
	for (size_t n = 0; n < nfilters; ++n) {
		model.filters()[n].convertTo(model.filters()[n], DataType<T>::type);
	}
	convolution_engine_->setFilters(model.filters());

	// initialize the tree of Parts
	parts_ = Parts(model.filters(), model.filtersi(), model.def(), model.defi(), model.bias(), model.biasi(),
			model.anchors(), model.biasid(), model.filterid(), model.defid(), model.parentid());

	// initialize the dynamic program
	dp_ = DynamicProgram<T>(model.thresh());

}



// declare all specializations of the template
template class PartsBasedDetector<float>;
template class PartsBasedDetector<double>;
