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

typedef std::vector<std::vector<cv::Mat> > vector2DMat;
typedef std::vector<std::vector<float> >   vector2Df;
typedef std::vector<std::vector<int> >     vector2Di;


class ComponentParts {
private:
	std::vector<cv::Mat>& filtersw_;
	std::vector<int>&     filtersi_;
	vector2Df&            defw_;
	std::vector<int>&     defi_;
	vector2Di&            biasid_;
	vector2Di&            filterid_;
	std::vector<int>&     parentid_;
	int                   self_;
public:
	ComponentParts(std::vector<cv::Mat>& filtersw, std::vector<int>& filtersi, vector2Df& defw, std::vector<int>& defi,
				   vector2Di& biasid, vector2Di& filterid, std::vector<int>& parentid, int self) :
	   filtersw_(filtersw), filtersi_(filtersi), filterid_(filterid_),
		defw_(defw), defi_(defi), biasid_(biasid),
		parentid_(parentid), self_(self) {}
	ComponentParts(const ComponentParts& other, int self) :
		filtersw_(other.filtersw_), filtersi_(other.filtersi_), filterid_(other.filterid_),
		defw_(other.defw_), defi_(other.defi_), biasid_(other.biasid_),
		parentid_(other.parentid_), self_(self) {}
	virtual ~ComponentParts() {}
	// assignment operator
	ComponentParts& operator=(const ComponentParts& rhs) {
		if (&rhs == this) return *this;
		filtersw_ = rhs.filtersw_;
		filtersi_ = rhs.filtersi_;
		defw_     = rhs.defw_;
		defi_     = rhs.defi_;
		biasid_   = rhs.biasid_;
		filterid_ = rhs.filterid_;
		parentid_ = rhs.parentid_;
		self_     = rhs.self_;
		return *this;
	}
	// get methods
	int nparts(void) const { return filterid_.size(); }
	int nmixtures(void) const { return filterid_[self_].size(); }
	int self(void) const { return self_; }
	// perform translation of indices internally
	const cv::Mat& filter(int mixture) const { assert(filterid_[self_].size() > mixture); return filtersw_[filterid_[self_][mixture]]; }
	const std::vector<cv::Mat> filters(void) const {
		std::vector<cv::Mat> out;
		for (int m = 0; m < filterid_[self_].size(); ++m) out.push_back(filtersw_[filterid_[self_][m]]);
		return out;
	}
	ComponentParts parent(void) const { return ComponentParts(*this, parentid_[self_]); }
	std::vector<ComponentParts> children(void) const {
		std::vector<ComponentParts> c;
		for (int n = self_; n < parentid_.size(); ++n) {
			ComponentParts cp(*this, n);
			if (parentid_[n] == self_) c.push_back(cp);
		}
		return c;
	}
};

class Parts {
private:
	std::vector<cv::Mat> filtersw_;
	std::vector<int>     filtersi_;
	vector2Df            defw_;
	std::vector<int>     defi_;
	std::vector<float>   biasw_;
	std::vector<int>     biasi_;
	vector2Df anchors_;
	std::vector<ComponentParts> components_;
public:
	Parts();
	virtual ~Parts();
};

#endif /* PARTS_HPP_ */
