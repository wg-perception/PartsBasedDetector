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
 *  File:    Parts.hpp
 *  Author:  Hilton Bristow
 *  Created: Jul 6, 2012
 */

#ifndef PARTS_HPP_
#define PARTS_HPP_
#include <assert.h>
#include "types.hpp"

/*! @class ComponentPart
 *  @brief Parts tree for a single component of the model
 *
 *  ComponentPart describes a tree of parts for a component of the model.
 *  The parts themselves are held within one monolithic pool, so
 *  the ComponentPart obfuscates the indexing into this pool
 */
class ComponentPart {
private:
	//! the filter weights
	vectorMat 	const * filtersw_;
	//! the filter indices
	vectori   	const * filtersi_;
	//! the bias weights
	vectorf   	const * biasw_;
	//! the bias indices
	vectori   	const * biasi_;
	//! the anchors (part position relative to parent)
	vectorPoint	const * anchors_;
	//! the deformation weights
	vector2Df 	const * defw_;
	//! the deformation indices
	vectori   	const * defi_;
	//! indexing schema for deformation (weights, indices and anchors)
	vector2Di 	const * defid_;
	//! indexing schema for the bias (weights and indices)
	vector2Di 	const * biasid_;
	//! indexing schema for the filters (weights and indices)
	vector2Di 	const * filterid_;
	//! indexing schema for the parent (for biasid_ and filterid_)
	vectori   	const * parentid_;
	//! the current part index
	int         	    self_;
public:
	//! default constructor, used for preallocating std::vectors, etc
	ComponentPart() {}
	//! internal constructor, called by Parts()
	ComponentPart(const vectorMat& filtersw, const vectori& filtersi,
			       const vectorf& biasw, const vectori& biasi, const vectorPoint& anchors,
			       const vector2Df& defw, const vectori& defi, const vector2Di& defid,
				   const vector2Di& biasid, const vector2Di& filterid, const vectori& parentid, int self) :
	   filtersw_(&filtersw), filtersi_(&filtersi),
	    biasw_(&biasw), biasi_(&biasi), anchors_(&anchors),
		defw_(&defw), defi_(&defi), defid_(&defid), biasid_(&biasid),
		filterid_(&filterid), parentid_(&parentid), self_(self) {}
	//! internal copy constructor, used to traverse to new nodes in the tree (parent, children or random access)
	ComponentPart(const ComponentPart& other, int self) :
		filtersw_(other.filtersw_), filtersi_(other.filtersi_),
		biasw_(other.biasw_), biasi_(other.biasi_), anchors_(other.anchors_),
		defw_(other.defw_), defi_(other.defi_), defid_(other.defid_), biasid_(other.biasid_),
		filterid_(other.filterid_), parentid_(other.parentid_), self_(self) {}
	//! destructor
	virtual ~ComponentPart() {}
	// get methods

	/*! @brief get the number of parts in this tree
	 *
	 * @return the number of parts
	 */
	size_t nparts(void) const { return filterid_->size(); }

	/*! @brief get the number of mixtures for the current part
	 *
	 * @return the number of mixtures. Note that not all components
	 * and parts will have the same number of mixtures
	 */
	size_t nmixtures(void) const { return (*filterid_)[self_].size(); }

	/*! @brief the current flattened tree index
	 *
	 * @return the current flattened tree index. self() == 0 is the root part
	 */
	int self(void) const { return self_; }

	// perform translation of indices internally
	/*! @brief return the filter for a part
	 *
	 * @param mixture the part mixture of interest
	 * @return the associated filter
	 */
	const cv::Mat& filter(size_t mixture = 0) const {
		assert((*filterid_)[self_].size() > mixture);
		return (*filtersw_)[(*filterid_)[self_][mixture]];
	}

	/*! @brief return the filters for all mixtures of a part
	 *
	 * @param out the filters to return
	 */
	void filters(vectorMat& out) const {
		out.clear();
		for (size_t m = 0; m < filterid_[self_].size(); ++m) {
			out.push_back((*filtersw_)[(*filterid_)[self_][m]]);
		}
	}
	/*! @brief the current part's parent
	 *
	 * @return a new ComponentPart which points to the parent of the current part
	 */
	ComponentPart parent(void) const { return ComponentPart(*this, (*parentid_)[self_]); }
	/*! @brief the current part's children
	 *
	 * @return a vector of ComponentParts, where each part's parent points to the current part
	 */
	std::vector<ComponentPart> children(void) const {
		std::vector<ComponentPart> c;
		for (size_t n = self_; (*parentid_)[n] <= self_; ++n) {
			ComponentPart cp(*this, n);
			if ((*parentid_)[n] == self_) c.push_back(cp);
		}
		return c;
	}
	/*! @brief the part score
	 *
	 * this method leverages the internal index translation to retrieve a score from an
	 * input vector of scores. The scores are assumed to have the same ordering as filtersw_
	 *
	 * @param scores the input vector of scores
	 * @param mixture the part mixture to retrieve
	 * @return the associated score for this part's mixture
	 */
	cv::Mat& score(vectorMat& scores, size_t mixture = 0) const {
		assert((*filterid_)[self_].size() > mixture);
		return scores[(*filterid_)[self_][mixture]];
	}
	//! the part's filter index
	int filteri(size_t mixture = 0) const { return (*filtersi_)[(*filterid_)[self_][mixture]]; }
	//! the part's bias
	vectorf bias(size_t mixture = 0) const {
		const int offset = (*biasid_)[self_][mixture];
		return vectorf(&((*biasw_)[offset]), &((*biasw_)[offset+nmixtures()]));
	}
	//! the part's bias index
	int biasi(size_t mixture = 0) const { return (*biasi_)[(*biasid_)[self_][mixture]]; }
	//! the part's deformation weights
	vectorf defw(size_t mixture = 0) const { return (*defw_)[(*defid_)[self_][mixture]]; }
	//! the part's deformation indices
	int defi(size_t mixture = 0) const { return (*defi_)[(*defid_)[self_][mixture]]; }
	//! the part's anchor (relative to its parent part)
	cv::Point anchor(size_t mixture = 0) const { return (*anchors_)[(*defid_)[self_][mixture]]; }
	//! the x size (width) of the part
	size_t xsize(size_t mixture = 0) const { return (*filtersw_)[(*filterid_)[self_][mixture]].rows; }
	//! the y size (height) of the part
	size_t ysize(size_t mixture = 0) const { return (*filtersw_)[(*filterid_)[self_][mixture]].rows; }
	//! is the current part the root
	bool isRoot(void) const { return self_ == 0; }
};

/*! @class Parts
 *  @brief a monolithic collection of part parameters
 *
 *  Parts describes a container class which holds a monolithic collection of part
 *  parameters. Its primary purpose is to hold concrete values of each parameter
 *  and spawn lightweight ComponentPart classes (which just have pointers to the
 *  same parameters) when object retrieval is required.
 *
 *  It is designed to maximally reflect the configuration of the Matlab implementation
 *  to avoid confusion
 */
class Parts {
private:
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
public:
	//! default constructor
	Parts() {}
	Parts(vectorMat& filtersw, vectori& filtersi, vector2Df& defw, vectori& defi, vectorf& biasw, vectori& biasi,
			vectorPoint& anchors, vector3Di& biasid, vector3Di& filterid, vector3Di& defid, vector2Di& parentid) :
				filtersw_(filtersw), filtersi_(filtersi), defw_(defw), defi_(defi), biasw_(biasw), biasi_(biasi),
				anchors_(anchors), biasid_(biasid), filterid_(filterid), defid_(defid), parentid_(parentid) {}
	//! default destructor
	virtual ~Parts() {}
	/*! @brief get a component of the model
	 *
	 * @param c the component to retrieve
	 * @param p the part to reference within that component (defaults to the root)
	 * @return the ComponentPart for component c at node p
	 */
	ComponentPart component(size_t c, size_t p = 0) {
		assert(c < biasid_.size() && c < filterid_.size() && c < parentid_.size());
		return ComponentPart(filtersw_, filtersi_, biasw_, biasi_, anchors_, defw_, defi_, defid_[c], biasid_[c], filterid_[c], parentid_[c], p);
	}
	//! the number of components in the model
	size_t ncomponents(void) const { return filterid_.size(); }
	/*! @brief the number of parts within a component
	 *
	 * @param c the component of interest
	 * @return the number of parts
	 */
	size_t nparts(size_t c) const {
		assert(c < biasid_.size() && c < filterid_.size() && c < parentid_.size());
		return filterid_[c].size();
	}
	//! all filters for all components and parts
	const vectorMat& filters(void) const { return filtersw_; }
};

#endif /* PARTS_HPP_ */
