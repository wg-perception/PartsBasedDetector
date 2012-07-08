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

typedef std::vector<int>                   vectori;
typedef std::vector<float>				   vectorf;
typedef std::vector<cv::Mat>               vectorMat;
typedef std::vector<std::vector<cv::Mat> > vector2DMat;
typedef std::vector<std::vector<float> >   vector2Df;
typedef std::vector<std::vector<int> >     vector2Di;
typedef std::vector<vector2Di>             vector3Di;


class ComponentPart {
private:
	vectorMat const * filtersw_;
	vectori   const * filtersi_;
	vectorf   const * biasw_;
	vectori   const * biasi_;
	vector2Df const * defw_;
	vectori   const * defi_;
	vector2Df const * anchors_;
	vector2Di const * biasid_;
	vector2Di const * filterid_;
	vectori   const * parentid_;
	int               self_;
public:
	ComponentPart() {}
	ComponentPart(const vectorMat& filtersw, const vectori& filtersi,
			       const vectorf& biasw, const vectori& biasi, const vector2Df& anchors,
			       const vector2Df& defw, const vectori& defi,
				   const vector2Di& biasid, const vector2Di& filterid, const vectori& parentid, int self) :
	   filtersw_(&filtersw), filtersi_(&filtersi), filterid_(&filterid),
	    biasw_(&biasw), biasi_(&biasi), anchors_(&anchors),
		defw_(&defw), defi_(&defi), biasid_(&biasid),
		parentid_(&parentid), self_(self) {}
	ComponentPart(const ComponentPart& other, int self) :
		filtersw_(other.filtersw_), filtersi_(other.filtersi_), filterid_(other.filterid_),
		biasw_(other.biasw_), biasi_(other.biasi_), anchors_(other.anchors_),
		defw_(other.defw_), defi_(other.defi_), biasid_(other.biasid_),
		parentid_(other.parentid_), self_(self) {}
	virtual ~ComponentPart() {}
	// get methods
	int nparts(void) const { return filterid_->size(); }
	int nmixtures(void) const { return filterid_[self_].size(); }
	int self(void) const { return self_; }
	// perform translation of indices internally
	const cv::Mat& filter(int mixture = 0) const {
		assert((*filterid_)[self_].size() > mixture);
		return (*filtersw_)[(*filterid_)[self_][mixture]]; }
	const vectorMat filters(void) const {
		vectorMat out;
		for (int m = 0; m < filterid_[self_].size(); ++m) {
			out.push_back((*filtersw_)[(*filterid_)[self_][m]]);
		}
		return out;
	}
	ComponentPart parent(void) const { return ComponentPart(*this, (*parentid_)[self_]); }
	std::vector<ComponentPart> children(void) const {
		std::vector<ComponentPart> c;
		for (int n = self_; n < parentid_->size(); ++n) {
			ComponentPart cp(*this, n);
			if ((*parentid_)[n] == self_) c.push_back(cp);
		}
		return c;
	}
	cv::Mat& score(vectorMat& scores, int mixture = 0) const {
		assert((*filterid_)[self_].size() > mixture);
		return scores[(*filterid_)[self_][mixture]];
	}
	int filteri(int mixture = 0) const { return (*filtersi_)[(*filterid_)[self_][mixture]]; }
	float bias(int mixture = 0) const { return (*biasw_)[(*biasid_)[self_][mixture]]; }
	int biasi(int mixture = 0) const { return (*biasi_)[(*biasid_)[self_][mixture]]; }
};

class Parts {
private:
	// common monolithic part components
	vectorMat filtersw_;
	vectori   filtersi_;
	vector2Df defw_;
	vectori   defi_;
	vectorf   biasw_;
	vectori   biasi_;
	vector2Df anchors_;
	// component accessors
	vector3Di biasid_;
	vector3Di filterid_;
	vector2Di parentid_;
public:
	Parts();
	virtual ~Parts();
	ComponentPart component(int n) const {
		assert(n < biasid_.size() && n < filterid_.size() && n < parentid_.size());
		return ComponentPart(filtersw_, filtersi_, biasw_, biasi_, anchors_, defw_, defi_, biasid_[n], filterid_[n], parentid_[n], 0);
	}
};

#endif /* PARTS_HPP_ */
