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
 *  File:    ConvolutionEngine.hpp
 *  Author:  Hilton Bristow
 *  Created: Oct 9, 2012
 */

#ifndef ICONVOLUTIONENGINE_HPP_
#define ICONVOLUTIONENGINE_HPP_

#include "types.hpp"

class IConvolutionEngine {
public:
	virtual ~IConvolutionEngine() {}

	/*! @brief probability density function
	 *
	 * A custom convolution-type operation for producing a map of probability density functions
	 * where each pixel indicates the likelihood of a positive detection
	 *
	 * @param features the input pyramid of features
	 * @param responses a 2D vector of pdfs, 1st dimension across scale, 2nd dimension across filter
	 */
	virtual void pdf(const vectorMat& features, vector2DMat& responses) = 0;

	/*! @brief set the convolve engine filters
	 *
	 * In many situations, the filters are static during operation of the detector
	 * so we can take advantage of some optimizations such as changing the memory layout
	 * of the filters, or shifting the filters to the GPU, etc. This function enables
	 * such a facility, and must necessarily be called before pdf()
	 *
	 * @param filters the vector of filters
	 */
	virtual void setFilters(const vectorMat& filters) = 0;
};



#endif /* ICONVOLUTIONENGINE_HPP_ */
