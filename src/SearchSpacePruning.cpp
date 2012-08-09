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
 *  File:    SearchSpacePruning.cpp
 *  Author:  Hilton Bristow
 *  Created: Aug 1, 2012
 */

#include "nms.hpp"
#include "SearchSpacePruning.hpp"
#include "Math.hpp"
#include <limits>
#include <iostream>
using namespace cv;
using namespace std;

template<typename T>
void SearchSpacePruning<T>::nonMaxSuppression(vector2DMat& rootv, const vectorf& scales) {

	const int N = rootv.size();
	const int C = rootv[0].size();
	/*
	// TODO: non-maxima suppression across all scales simultaneously,
	// so good detections at different scales cannot be overlaid
	const Size maxsize = rootv[0][0].size();
	vectorMat scaled(N*C);
	for (int n = 0; n < N; ++n) {
		for (int c = 0; c < C; ++c) {
			Mat resized;
			resize(rootv[n][c], resized, maxsize);
			scaled[n*C+c] = resized;
		}
	}
	Mat maxv, maxi;
	Math::reduceMax<T>(scaled, maxv, maxi);
	*/


	T smallest;
	if (numeric_limits<T>::is_signed) {
		smallest = -numeric_limits<T>::max();
	} else {
		smallest = 0;
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int nc = 0; nc < N * C; ++nc) {
		const int n = nc / C;
		const int c = nc % C;
		Mat maxima;
		nonMaximaSuppression(rootv[n][c], scales[n] * 5, maxima);
		rootv[n][c].setTo(smallest, maxima == 0);
	}
}

template<typename T>
void SearchSpacePruning<T>::filterResponseByDepth(vector2DMat& pdfs, const vector<Size>& fsizes, const Mat& depth, const vectorf& scales, const float X, const float fx) {

	const int N  = pdfs.size();
	const int F  = pdfs[0].size();

#ifdef _OPENMP
	#pragma omp parallel for
#endif
	for (int nf = 0; nf < N*F; ++nf) {
		const int n = nf / F;
		const int f = nf % F;

		// create a mask of plausible depths given the object size
		// and the scale of the image
		Mat sdepth;
		resize(depth, sdepth, pdfs[n][f].size());

		// calculate the depth of the part in real-world coordinates,
		// given the 3d width of the part, the focal length of the camera,
		// and the width of the part in the image
		float Z = fx*X/scales[n];
	}
}

template<typename T>
void SearchSpacePruning<T>::filterCandidatesByDepth(Parts& parts, vectorCandidate& candidates, const Mat& depth, const float zfactor) {

	vectorCandidate new_candidates;
	const int N = candidates.size();
	for (int n = 0; n < N; ++n) {
		const int c = candidates[n].component();
		const int nparts = parts.nparts(c);
		const vector<Rect>& boxes = candidates[n].parts();
		for (int p = nparts-1; p >= 1; --p) {
			ComponentPart part = parts.component(c,p);
			Point anchor = part.anchor(0);
			Rect child   = boxes[part.self()];
			Rect parent  = boxes[part.parent().self()];
			T cmed_depth = Math::median<T>(depth(child));
			T pmed_depth = Math::median<T>(depth(parent));
			if (cmed_depth > 0 && pmed_depth > 0) {
				if (abs(cmed_depth-pmed_depth) > norm(anchor)*zfactor) break;
			}
			if (p == 1) new_candidates.push_back(candidates[n]);
		}
	}
	candidates = new_candidates;
}

// declare all specializations of the template (this must be the last declaration in the file)
template class SearchSpacePruning<float>;
template class SearchSpacePruning<double>;
