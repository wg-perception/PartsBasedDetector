/*
 * SpatialConvolutionEngine.cpp
 *
 *  Created on: Oct 9, 2012
 *      Author: hiltonbristow
 */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "SpatialConvolutionEngine.hpp"
using namespace std;
using namespace cv;

SpatialConvolutionEngine::SpatialConvolutionEngine(int type, unsigned int flen) :
	type_(type), flen_(flen) {}

SpatialConvolutionEngine::~SpatialConvolutionEngine() {
	// TODO Auto-generated destructor stub
}

/*! @brief Convolve two matrices, with a stride of greater than one
 *
 * This is a specialized 2D convolution algorithm with a stride of greater
 * than one. It is designed to convolve a filter with a feature, where at
 * each pixel an SVM must be evaluated (leading to a stride of SVM weight length).
 * The convolution can be thought of as flattened a 2.5D convolution where the
 * (i,j) dimension is the spatial plane and the (k) dimension is the SVM weights
 * of the pixels.
 *
 * The function supports multithreading via OpenMP
 *
 * @param feature the feature matrix
 * @param filter the filter (SVM)
 * @param pdf the response to return
 * @param stride the SVM weight length
 */
void SpatialConvolutionEngine::convolve(const Mat& feature, vectorFilterEngine& filter, Mat& pdf, const unsigned int stride) {

	// error checking
	assert(feature.depth() == type_);

	// split the feature into separate channels
	vectorMat featurev;
	split(feature.reshape(stride), featurev);

	// calculate the output
	Rect roi(0,0,-1,-1); // full image
	Point offset(0,0);
	Size fsize = featurev[0].size();
	pdf = Mat::zeros(fsize, type_);

	for (unsigned int c = 0; c < stride; ++c) {
		Mat pdfc(fsize, type_);
		filter[c]->apply(featurev[c], pdfc, roi, offset, true);
		pdf += pdfc;
	}
}


/*! @brief Calculate the responses of a set of features to a set of filter experts
 *
 * A response represents the likelihood of the part appearing at each location of
 * the feature map. Parts are support vector machines (SVMs) represented as filters.
 * The convolution of a filter with a feature produces a probability density function
 * (pdf) of part location
 * @param features the input features (at different scales, and by extension, size)
 * @param responses the vector of responses (pdfs) to return
 */
void SpatialConvolutionEngine::pdf(const vectorMat& features, vector2DMat& responses) {

	// preallocate the output
	const unsigned int M = features.size();
	const unsigned int N = filters_.size();
	responses.resize(M, vectorMat(N));
	// iterate
#ifdef _OPENMP
	omp_set_num_threads(8);
	#pragma omp parallel for
#endif
	for (unsigned int n = 0; n < N; ++n) {
		for (unsigned int m = 0; m < M; ++m) {
			Mat response;
			convolve(features[m], filters_[n], response, flen_);
			responses[m][n] = response;
		}
	}
}

/*! @brief set the filters
 *
 * given a set of filters, split each filter channel into a plane,
 * in preparation for convolution
 *
 * @param filters the filters
 */
void SpatialConvolutionEngine::setFilters(const vectorMat& filters) {

	const unsigned int N = filters.size();
	filters_.clear();
	filters_.resize(N);

	// split each filter into separate channels, and create a filter engine
	const unsigned int C = flen_;
	for (unsigned int n = 0; n < N; ++n) {
		vectorMat filtervec;
		std::vector<Ptr<FilterEngine> > filter_engines(C);
		split(filters[n].reshape(C), filtervec);

		// the first N-1 filters have zero-padding
		for (unsigned int m = 0; m < C-1; ++m) {
			Ptr<FilterEngine> fe = createLinearFilter(type_, type_,
					filtervec[m], Point(-1,-1), 0, BORDER_CONSTANT, -1, Scalar(0,0,0,0));
			filter_engines[m] = fe;
		}

		// the last filter has one-padding
		Ptr<FilterEngine> fe = createLinearFilter(type_, type_,
				filtervec[C-1], Point(-1,-1), 0, BORDER_CONSTANT, -1, Scalar(1,1,1,1));
		filter_engines[C-1] = fe;
		filters_[n] = filter_engines;
	}
}
