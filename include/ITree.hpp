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
 *  File:    ITree.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 28, 2012
 */

#ifndef ITREE_HPP_
#define ITREE_HPP_
#include <vector>

/*! @class ITree
 *  @brief Read only Tree Interface
 *
 *  This class defines a simple read-only tree interface. Invariants:
 *  (1) Once the tree is constructed, it cannot be modified
 *  (2) The tree can only be constructed from the leaves to the root
 *  since a node constructor requires knowledge of the node children.
 *  A head recursive constructor is a graceful way to construct the
 *  tree
 */
template<class T>
class ITree {
protected:
	ITree();
public:
	virtual ~ITree() = 0;
	virtual const std::vector<ITree&> children(void) const = 0;
	virtual const int level(void) const = 0;
	virtual const int ndescendants(void) const = 0;
	virtual const T& value(void) const = 0;
	virtual const bool isLeaf(void) const = 0;
};

ITree::~ITree(void) {}
#endif /* ITREE_HPP_ */
