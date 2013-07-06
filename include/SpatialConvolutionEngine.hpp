/*
 * SpatialConvolutionEngine.hpp
 *
 *  Created on: Oct 9, 2012
 *      Author: hiltonbristow
 */

#ifndef SPATIALCONVOLUTIONENGINE_HPP_
#define SPATIALCONVOLUTIONENGINE_HPP_

#include "IConvolutionEngine.hpp"

class SpatialConvolutionEngine: public IConvolutionEngine {
private:
	//! the internally supported convolution type, taken from the filter type
	int type_;
	//! the number of layers to each filter
	unsigned int flen_;
	//! the internal representation of the filters
	vector2DFilterEngine filters_;
	void convolve(const cv::Mat& feature, vectorFilterEngine& filter, cv::Mat& pdf, const unsigned int stride);
public:
	SpatialConvolutionEngine(int type, unsigned int flen);
	virtual ~SpatialConvolutionEngine();
	virtual void setFilters(const vectorMat& filters);
	virtual void pdf(const vectorMat& features, vector2DMat& responses);
};

#endif /* SPATIALCONVOLUTIONENGINE_HPP_ */
