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
 *  File:    Model.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef MODEL_HPP_
#define MODEL_HPP_
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "types.hpp"

/*! @class Model
 *  @brief Monolithic container class for storing model parameters
 */
class Model {
protected:
	// member variables
	// common monolithic part components
	//! the filter weights
	vectorMat 	filtersw_;
	//! the filer indices
	vectori   	filtersi_;
	//! the deformation weights
	vector2Df 	defw_;
	//! the deformation indices
	vectori   	defi_;
	//! the bias weights
	vectorf   	biasw_;
	//! the bias indices
	vectori   	biasi_;
	//! the anchors (part position relative to parent)
	vectorPoint anchors_;
	// component accessors
	//! indexing schema for the bias (weights and indices)
	vector3Di 	biasid_;
	//! indexing schema for the filters (weights and indices)
	vector3Di 	filterid_;
	//! indexing schema for the deformation (weights, indices and anchors
	vector3Di 	defid_;
	//!indexing schema for the parent (for biasid_, filterid_ and defid_)
	vector2Di 	parentid_;
	//! a unique string identifier for the model
	std::string name_;
	//! the connectivity of the parts, where each element is a reference to the part's parent
	vectori conn_;
	//! the number of parts
	int nparts_;
	//! the number of mixtures
	int nmixtures_;
	//! the number of scales at which to compute features
	int nscales_;
	//! the threshold for a positive detection
	float thresh_;
	//! the spatial pooling size when computing features
	int binsize_;
	//! the length of the feature vector in each bin
	int flen_;
	//! the number of orientations per HOG feature bin
	int norient_;

public:
	Model() {}
	virtual ~Model() {}
	vectorMat& filters(void) { return filtersw_; }
	vectori& filtersi(void) { return filtersi_; }
	vector2Df& def(void) { return defw_; }
	vectori& defi(void) { return defi_; }
	vectorf& bias(void) { return biasw_; }
	vectori& biasi(void) { return biasi_; }
	vectorPoint& anchors(void) { return anchors_; }
	vector3Di& filterid(void) { return filterid_; }
	vector3Di& biasid(void) { return biasid_; }
	vector3Di& defid(void) { return defid_; }
	vector2Di& parentid(void) { return parentid_; }
	std::string name(void) { return name_; }
	vectori& conn(void) { return conn_; }
	int nparts(void) const { return nparts_; }
	int nmixtures(void) const { return nmixtures_; }
	float thresh(void) const { return thresh_; }
	int binsize(void) const { return binsize_; }
	int nscales(void) const { return nscales_; }
	int flen(void) const { return flen_; }
	int norient(void) const { return norient_; }
	int ncomponents(void) const { return filterid_.size(); }

	virtual bool serialize(const std::string& filename) const = 0;
	virtual bool deserialize(const std::string& filename) = 0;
};

#endif /* MODEL_HPP_ */
