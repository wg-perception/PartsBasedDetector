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
 *  File:    HOGFeatures.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "HOGFeatures.hpp"
using namespace std;
using namespace cv;

/*! @brief Calculate features at multiple scales
 *
 * Features are calculated first at native resolution,
 * then progressively downsampled to coarser spatial
 * resolutions
 *
 * @param im the input image at native resolution
 * @return the pyramid of features, fine to coarse, each calculated via
 * features()
 */
std::vector<cv::Mat> HOGFeatures::pyramid(const cv::Mat& im) {

	vector<Mat> pyrafeatures;
	scales_.clear();

	// calculate the scaling factorA
	Size imsize = im.size();
	float sc     = exp( floor( log ( (float)min(im.rows, im.cols)/(30.0f*(float)binsize_) ) )/(nscales_ - 1) );
	int interval = round( 1.0f / log2(sc) );

	// perform the non-power of two scaling
	Mat scaled;
	Mat padded;
	Mat feature;
	for (int i = 0; i < interval; ++i) {
		resize(im, scaled, imsize * (1.0f/pow(sc,i)));
		feature = features(scaled);
		copyMakeBorder(feature, padded, 1, 1, flen_, flen_, BORDER_CONSTANT, 1);
		pyrafeatures.push_back(padded);
		scales_.push_back(1.0f/pow(sc,i));

		// perform the subsequent power of two scaling
		for (int j = i+interval; j < nscales_; j+=interval) {
			scaled = pyrDown(scaled, scaled);
			feature = features(scaled);
			copyMakeBorder(feature, padded, 1, 1, flen_, flen_, BORDER_CONSTANT, 1);
			pyrafeatures.push_back(padded);
			scales_.push_back(0.5 * scales_[j-interval]);
		}
	}

	return pyrafeatures;
}

/*! @brief compute the HOG features for an image
 *
 * This method computes the HOG features for an image, given the
 * binsize_ and flen_ class members. The output is effectively a
 * 3D matrix (i,j,k) that has been flattened to a 2D (i,j*k) matrix
 * for faster processing. The (i,j) dimensions represent the resultant
 * spatial size of the response (ie im.size() / binsize_) and the
 * (k) dimension represents the histogram weights (length flen_)
 *
 * The function supports multithreading via OpenMP
 *
 * @param im the input image (must be color of type CV_8UC3)
 * @return the HOG features as a 2D matrix
 */
Mat HOGFeatures::features(const Mat& im) const {

	// compute the size of the output matrix
	Size imsize = im.size();
	Size blocks = imsize;
	Size outsize = imsize;
	blocks.height = floor(blocks.height / binsize_);
	blocks.width  = floor(blocks.width  / binsize_);
	outsize.width = outsize.width * flen_;

	Mat histm = Mat::zeros(outsize, CV_32FC1);
	Mat normm = Mat::zeros(blocks, CV_32FC1);

	// eps to avoid division by zero
	const double eps = 0.0001;

	// unit vectors to compute gradient orientation
	double uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
	double vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

	// calculate the zero offset
	Size offset((imsize.height - blocks.height*binsize_)/2, (imsize.width - blocks.width*binsize_)/2);
	const float* src = im.ptr<float>(0);
	float* hist = histm.ptr<float>(0);
#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
#endif
	for (int i = 1; i < blocks.height-1; ++i) {
		for (int j = 1; j < blocks.width-1; ++j) {

			// blue color channel
			const float* s = src + (i+offset.height)*imsize.height + (j+offset.width);
			float dy = *(s+imsize.width) - *(s-imsize.width);
			float dx = *(s+1) - *(s-1);
			float  v = dx*dx + dy*dy;

			// green color channel
			s += imsize.width*imsize.height;
			float dyg = *(s+imsize.width) - *(s-imsize.width);
			float dxg = *(s+1) - *(s-1);
			float  vg = dxg*dxg + dyg*dyg;

			// red color channel
			s += imsize.width*imsize.height;
			float dyr = *(s+imsize.width) - *(s-imsize.width);
			float dxr = *(s+1) - *(s-1);
			float  vr = dxr*dxr + dyr*dyr;

			// pick the channel with the strongest gradient
			if (vg > v) { v = vg; dx = dxg; dy = dyg; }
			if (vr > v) { v = vr; dx = dxr; dy = dyr; }

			// snap to one of 18 orientations
			float best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; ++o) {
				float dot = uu[o]*dx + vv[o]*dy;
				if (dot > best_dot) { best_dot = dot; best_o = o; }
				else if (-dot > best_dot) { best_dot = -dot; best_o = o+9; }
			}

			// add to 4 histograms around pixel using linear interpolation
			float ip = ((float)i+0.5)/(float)binsize_ - 0.5;
			float jp = ((float)j+0.5)/(float)binsize_ - 0.5;
			int iip = (int)floor(ip);
			int ijp = (int)floor(jp);
			float vi0 = ip-iip;
			float vj0 = jp-ijp;
			float vi1 = 1.0-vi0;
			float vj1 = 1.0-vj0;
			v = sqrt(v);

			if (iip >= 0 && ijp >= 0) *(hist + (iip*blocks.width + ijp)*flen_ + best_o) = vi1*vj1*v;
			if (iip >= 0 && ijp <= blocks.width) *(hist + (iip*blocks.width + ijp+1)*flen_ + best_o) = vj1*vi0*v;
			if (iip <= blocks.height && ijp >= 0) *(hist + ((iip+1)*blocks.width + ijp)*flen_ + best_o) = vi1*vj0*v;
			if (iip <= blocks.height && ijp <= blocks.width) *(hist + ((iip+1)*blocks.width + ijp+1)*flen_ + best_o) = vi1*vj1*v;
		}
	}

	// compute the energy in each block by summing over orientations
	float* norm = normm.ptr<float>(0);
	for (int i = 0; i < blocks.height*blocks.width; ++i) {
		for (int o = 0; o < 9; ++o) norm[i] += pow(hist[i*flen_+o] + hist[i*flen_+o+9], 2);
	}

	return histm;
}

/*! @brief Convolve two matrices, with a stride of greater than one
 *
 * This is a specialized 2D convolution algorithm with a stride of greater
 * than one. It is designed to convolve a filter with a feature, where at
 * each pixel an SVM must be evaluated (leading to a stride of SVM weight length).
 * The convolution can be thought of as flattened a 2.5D convolution where the
 * (i,j) dimension is the spatial plane and the (k) dimension is the SVM weights
 * of the pixels. As one would expect, this method is slow
 *
 * The function supports multithreading via OpenMP
 *
 * @param feature the feature matrix
 * @param filter the filter (SVM)
 * @param stride the SVM weight length
 * @return the response (pdf)
 */
template<typename T>
Mat HOGFeatures::convolve(const Mat& feature, const Mat& filter, int stride) {

	// error checking
	assert(feature.type() == feature.type());
	assert(feature.cols % stride == 0 && filter.cols % stride == 0);

	// really basic convolution algorithm with a stride greater than one
	const int M = feature.rows - filter.rows + 1;
	const int N = feature.cols - filter.cols + stride;
	const int H = filter.rows;
	const int W = filter.cols;
	Mat response(Size(M, N), feature.type());
	const T* feat_ptr = feature.ptr<T>(0);
	const T* filt_ptr = filter.ptr<T>(0);
	const T* filt_start = filter.ptr<T>(0);
	T* resp_ptr = response.ptr<T>(0);
#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
#endif
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; n+=stride) {
			T accum = 0;
			filt_ptr = filt_start;
			feat_ptr = feature.ptr<T>(m) + n;
			for (int h = 0; h < H; ++h) {
				while (filt_ptr < filt_start+h*W) {
					accum += *(filt_ptr++) * *(feat_ptr++);
				}
				feat_ptr = feature.ptr<T>(m+h) + n;
			}
			*(resp_ptr++) = accum;
		}
	}
	return response;
}

/*! @brief Calculate the responses of a set of features to a set of filter experts
 *
 * A response represents the likelihood of the part appearing at each location of
 * the feature map. Parts are support vector machines (SVMs) represented as filters.
 * The convolution of a filter with a feature produces a probability density function
 * (pdf) of part location
 * @param features the input features (at different scales, and by extension, size)
 * @param filters the filters representing the parts across mixtures
 * @return a vector of responses (pdfs)
 */
vector<Mat> HOGFeatures::pdf(const vector<Mat>& features, const vector<Mat>& filters) {

	// preallocate the output
	int M = features.size();
	int N = filters.size();
	vector<Mat> responses;
	responses.reserve(M*N);

	// iterate
	Mat feature = features[0];
	Mat filter = filters[0];
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			Mat response = convolve<float>(feature, filter, flen_);
			responses.push_back(response);
		}
	}
	return responses;

}
