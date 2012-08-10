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
 *  File:    Part.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 28, 2012
 */

#include "Part.hpp"
using namespace std;
using namespace cv;




void Part::toVector(vectorMat& vec) {
	// if root, allocate space for all of the filters
	if (isRoot()) vec.resize((ndescendants_+1) * nmixtures_);

	// add my filters to the vector, then my children's, etc
	int os = pos_ * nmixtures_;
	for (unsigned int n = 0; n < nmixtures_; ++n) vec[os+n] = filters_[n];
	for (unsigned int c = 0; c < children_.size(); ++c) children_[c].toVector(vec);
}


static vectori find(vectori vals, int val) {
	vectori idx;
	for (unsigned int n = 0; n < vals.size(); ++n) if(vals[n] == val) idx.push_back(n);
	return idx;
}

static Part constructPartHierarchyRecursive(vector2DMat& filters, vectori& parents, int level, int self) {

	// find all of the children who have self (current Part) as a parent
	vectori cidx = find(parents, self);
	std::vector<Part> children;
	int ndescendants = 0;
	vector2Df bias;

	// recursive termination criteria:
	// this Part has no children. We are at a leaf node
	if (cidx.size() == 0) return Part(bias, filters[self], self, children, level, ndescendants);

	// otherwise, recursively create child Parts
	for (unsigned int n = 0; n < cidx.size(); ++n) {
		Part child = constructPartHierarchyRecursive(filters, parents, level+1, cidx[n]);
		ndescendants += (child.ndescendants()+1);
		children.push_back(child);
	}
	return Part(bias, filters[self], self, children, level, ndescendants);
}


/*! @brief construct a tree of Parts
 *
 * Given a set of Part components (filters, parents, w, bias), recursively construct
 * a tree of Parts
 * @param filters
 * @param parents
 * @return
 */
Part Part::constructPartHierarchy(vector2DMat& filters, vectori& parents) {

	// error checking
	assert(filters.size() == parents.size());

	// construct the Part tree, from the root node
	return constructPartHierarchyRecursive(filters, parents, 0, 0);

}
