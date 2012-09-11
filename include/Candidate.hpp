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
 *  File:    Candidate.hpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#ifndef CANDIDATE_HPP_
#define CANDIDATE_HPP_
#include <algorithm>
#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include "types.hpp"
#include "Rect3.hpp"
#include "Math.hpp"

/*! @class Candidate
 *  @brief detection candidate
 *
 * Candidate describes a storage class for object detection candidates. A single Candidate
 * represents a detection for a tree of parts. The candidate is parameterized by a cv::Rect
 * bounding box and detection confidence for each part
 */
class Candidate {
private:
	//! the bounding boxes of the parts
	std::vector<cv::Rect> parts_;
	//! the confidence scores of the parts
	vectorf confidence_;
	//! the model component the candidate belongs to
	int component_;
public:
	Candidate() {}
	virtual ~Candidate() {}
	//! return the vector of parts
	const std::vector<cv::Rect>& parts(void) const { return parts_; }
	//! return the vector of confidence scores
	const vectorf& confidence(void) const { return confidence_; }
	//! add a part score to the candidate, parameterized by a bounding box and confidence value
	void addPart(cv::Rect r, float confidence) { parts_.push_back(r); confidence_.push_back(confidence); }
	//! get the root score of the detection. Using for sorting
	float score(void) const { return (confidence_.size() > 0) ? confidence_[0] : -std::numeric_limits<double>::infinity(); }
	//! set the root score of the detection
	void setScore(float confidence) { if (confidence_.size() == 0) confidence_.resize(1); confidence_[0] = confidence; }
	//! set the candidate component
	void setComponent(int c) { component_ = c; }
	//! get the candidate component
	int component(void) { return component_; }
	//! rescale the parts
	void resize(const float factor) {
		for (unsigned int n = 0; n < parts_.size(); ++n) {
			parts_[n].height *= factor;
			parts_[n].width  *= factor;
			parts_[n].y      *= factor;
			parts_[n].x      *= factor;
		}
	}
	//! descending comparison method for ordering objects of type Candidate
	static bool descending(Candidate c1, Candidate c2) { return c1.score() > c2.score(); }

	/*! @brief Sort the candidates from best to worst, in place
	 *
	 * @param candidates the vector of candidates to sort
	 */
	static void sort(vectorCandidate& candidates) {
		std::sort(candidates.begin(), candidates.end(), descending);
	}

	/*! @brief create a single bounding box around the detection taken from the part limit
	 *
	 * @return a single bounding Rect
	 */
	cv::Rect boundingBox(void) const {
		cv::Rect hull = parts_[0];
		for (unsigned int n = 0; n < parts_.size(); ++n) {
			hull = hull | parts_[n];
		}
		return hull;
	}

	/*! @brief create a single bounding box around the detection from mean and standard deviation
	 *
	 * @return a bounding box
	 */
	cv::Rect boundingBoxNorm(void) const {
		const unsigned int nparts = parts_.size();
		cv::Mat_<int> xpts(cv::Size(1,nparts));
		cv::Mat_<int> ypts(cv::Size(1,nparts));
		for (unsigned int n = 0; n < nparts; ++n) {
			const cv::Point centroid = (parts_[n].tl() + parts_[n].br())*0.5;
			xpts(n) = centroid.x;
			ypts(n) = centroid.y;
		}
		cv::Scalar xmean, ymean, xstd, ystd;
		cv::meanStdDev(xpts, xmean, xstd);
		cv::meanStdDev(ypts, ymean, ystd);
		return cv::Rect(xmean(0)-1.5*xstd(0), ymean(0)-1.5*ystd(0), 3*xstd(0), 3*ystd(0));
	}

	/*! @brief create a single bounding box in 3D
	 *
	 * Given a 3D image, return
	 * @param depth
	 * @return
	 */
	Rect3 boundingBox3D(cv::Mat& depth) const {
		const unsigned int nparts = parts_.size();
		cv::Point3_<int> minv(1,1,1); minv *=  std::numeric_limits<int>::max();
		cv::Point3_<int> maxv(1,1,1); maxv *= -std::numeric_limits<int>::min();
		for (unsigned int n = 0; n < nparts; ++n) {
			int med;
			switch (depth.depth()) {
				case CV_16U: med = Math::median<uint16_t>(depth(parts_[n])); break;
				case CV_32F: med = Math::median<float>(depth(parts_[n]))*1000; break;
				case CV_64F: med = Math::median<double>(depth(parts_[n]))*1000; break;
			}
			const cv::Rect r = parts_[n];
			if (r.x + r.width/2  < minv.x) minv.x = r.x + r.width/2;
			if (r.y + r.height/2 < minv.y) minv.y = r.y + r.height/2;
			if (r.x + r.width/2  > maxv.x) maxv.x = r.x + r.width/2;
			if (r.y + r.height/2 < maxv.y) maxv.y = r.y + r.height/2;

			if (med < minv.z) minv.z = med;
			if (med > maxv.z) maxv.z = med;
		}
		return Rect3(minv, maxv);
	}

	/*! @brief suppress non-maximal candidates
	 *
	 * Given a vector of candidates, keep only the maximal candidates which
	 * overlap less than a defined fractional area. If overlap is 0.0 (the
	 * default value) no overlap is allowed. If, for example, the overlap is
	 * 0.2, then two candidates' bounding boxes can intersect by 20%
	 *
	 * @param im the input image from which the candidates were found
	 * @param candidates the vector of candidates
	 * @param overlap the allowable overlap [0.0 1.0)
	 */
	static void nonMaximaSuppression(const cv::Mat& im, vectorCandidate& candidates, const float overlap=0.0f) {

		// create a scratch space that we can draw on
		const unsigned int N = candidates.size();
		cv::Mat scratch = cv::Mat::zeros(im.size(), CV_8U);
		cv::Rect bounds = cv::Rect(0,0,0,0) + im.size();

		// the current insertion position in the vector
		unsigned int keep = 0;

		/* iterate through the boxes, checking:
		 * 1) has the area under the box been painted?
		 * 2) if so, is it under the threshold?
		 * 3) if so, keep this box and paint the area
		 * 4) repeat
		 */
		for (unsigned int n = 0; n < N; ++n) {
			cv::Rect box = candidates[n].boundingBox() & bounds;
			cv::Scalar boxsum = sum(scratch(box));
			if (boxsum[0] / box.area() > overlap) continue;
			scratch(box) = 1;
			candidates[keep] = candidates[n];
			keep++;
		}

		// simply delete the trailing end of the candidates
		candidates.resize(keep);
	}

	/*! @brief return a masked representation of a set of candidates
	 *
	 * Given a vector of candidates which have already been non-maximally
	 * suppressed, return an image mask where zero values represent
	 * regions which do not contain objects, and all integer values
	 * represent unique object locations.
	 *
	 * I.e. object_7 = (mask == 7) returns a mask where nonzero elements
	 * bound the 7th best detection
	 *
	 * @param im the input image from which the candidates were found
	 * @param candidates the vector of candidates
	 * @param mask a mask of type CV_8U the same size as im
	 */
	static void mask(const cv::Mat& im, const vectorCandidate& candidates, cv::Mat& mask) {

		// allocate the mask
		const unsigned int N = candidates.size();
		mask = cv::Mat::zeros(im.size(), CV_8U);
		cv::Rect bounds = cv::Rect(0,0,0,0) + im.size();

		for (unsigned int n = 0; n < N; ++n) {
			cv::Rect box = candidates[n].boundingBox() & bounds;
			mask(box).setTo(n+1, mask(box) == 0);
		}
	}

};

#endif /* CANDIDATE_HPP_ */
