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
		for (size_t n = 0; n < parts_.size(); ++n) {
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
		for (size_t n = 0; n < parts_.size(); ++n) {
			hull = hull | parts_[n];
		}
		return hull;
	}

	/*! @brief create a single bounding box around the detection from mean and standard deviation
	 *
	 * @return a bounding box
	 */
	cv::Rect boundingBoxNorm(void) const {
		const size_t nparts = parts_.size();
		cv::Mat_<int> xpts(cv::Size(1,nparts));
		cv::Mat_<int> ypts(cv::Size(1,nparts));
		for (size_t n = 0; n < nparts; ++n) {
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
	 * Given an image, its depth correspondences and a candidate,
	 * return an approximate 3D bounding box which encapsulates the object
	 * @param im the color image
	 * @param depth the depth image (may be of different resolution to the color image)
	 * @return
	 */
	Rect3d boundingBox3D(const cv::Mat& im, const cv::Mat& depth) const {

		const size_t nparts = parts_.size();
		const cv::Rect bounds = cv::Rect(0,0,0,0) + im.size();
		const cv::Rect bb  = this->boundingBox();
		const cv::Rect bbn = this->boundingBoxNorm();

		cv::Size_<double> imsize = im.size();
		cv::Size_<double> dsize  = depth.size();
		cv::Point_<double> s = cv::Point_<double>(dsize.width / imsize.width, dsize.height / imsize.height);

		cv::Mat_<float> points;
		std::vector<cv::Rect> boxes;
		for (size_t n = 0; n < nparts; ++n) {
			// only keep the intersection of the part with the image frame
			boxes.push_back(parts_[n] & bounds);
		}
		boxes.push_back(bbn & bounds);

		for (size_t n = 0; n < boxes.size(); ++n) {
			// scale the part down to match the depth image size
			cv::Rect& r = boxes[n];
			r.x = r.x * s.x;
			r.y = r.y * s.y;
			r.width  = r.width  * s.x;
			r.height = r.height * s.y;

			// add the valid points
			cv::Mat_<float> part = depth(r);
			if(part.empty())
				continue;

			for (cv::MatIterator_<float> it = part.begin(); it != part.end(); ++it) {
				if (*it != 0 && !std::isnan(*it)) points.push_back(*it);
			}

			if(points.empty())
			{
				return Rect3d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
						0, 0, 0);
			}

		}

		// sort the points
		std::sort(points.begin(), points.end());
		cv::resize(points, points, cv::Size(1, 400));

		// get the median of the points
		const size_t M = points.rows;
		const size_t midx = M/2;
		float median = points[midx][0];

		// filter the points
		cv::Mat_<float> g, dog, dpoints;
		cv::Matx<float, 3, 1> diff(-1, 0, 1);
		g= cv::getGaussianKernel(35, 4, CV_32F);
		cv::filter2D(g, dog, -1, diff);
		cv::filter2D(points, dpoints, -1, dog);

		// starting at the median point, walk up and down until a gradient threshold (0.1)
		size_t dminidx = midx, dmaxidx = midx;
		for (size_t m = midx; m < M; ++m) {
			if (fabs(dpoints[m][0]) > 0.035) break;
			dmaxidx = m;
		}
		for (int m = midx; m >= 0; --m) {
			if (fabs(dpoints[m][0]) > 0.035) break;
			dminidx = m;
		}

		// construct the 3D bounding box
		cv::Point3_<double> tl(bb.x,      bb.y,      points[dminidx][0]);
		cv::Point3_<double> br(bb.br().x, bb.br().y, points[dmaxidx][0]);

		return Rect3d(tl, br);
	}
/*

		const size_t nparts = parts_.size();
		cv::Size_<double> imsize = im.size();
		cv::Size_<double> dsize  = depth.size();
		const cv::Rect bounds = cv::Rect(0,0,0,0) + im.size();
		cv::Point_<double> s = cv::Point_<double>(dsize.width / imsize.width, dsize.height / imsize.height);
		cv::Point3_<double> minv(1,1,1); minv *=  std::numeric_limits<double>::max();
		cv::Point3_<double> maxv(1,1,1); maxv *= -std::numeric_limits<double>::max();
		for (size_t n = 0; n < nparts; ++n) {
			double med;
			// only keep the intersection of the part with the image frame
			cv::Rect r = parts_[n] & bounds;

			// scale the part down to match the depth image size
			r.x = r.x * s.x;
			r.y = r.y * s.y;
			r.width  = r.width  * s.x;
			r.height = r.height * s.y;

			switch (depth.depth()) {
				case CV_16U: med = Math::median<uint16_t>(depth(r)); break;
				case CV_32F: med = Math::median<float>(depth(r)); 	 break;
				case CV_64F: med = Math::median<double>(depth(r)); 	 break;
			}

			if (r.x < minv.x) minv.x = r.x;
			if (r.y < minv.y) minv.y = r.y;
			if (r.x + r.width  > maxv.x) maxv.x = r.x + r.width;
			if (r.y + r.height > maxv.y) maxv.y = r.y + r.height;

			if (med < minv.z) minv.z = med;
			if (med > maxv.z) maxv.z = med;
		}

		// scale the parts back up to match the color image size
		minv.x = minv.x / s.x;
		minv.y = minv.y / s.y;
		maxv.x = maxv.x / s.x;
		maxv.y = maxv.y / s.y;

		// extrapolate some depth
		minv.z = minv.z - (maxv.z-minv.z);
		maxv.z = maxv.z + (maxv.z-minv.z);

		return Rect3d(minv, maxv);
	}
	*/

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
		const size_t N = candidates.size();
		const cv::Rect bounds = cv::Rect(0,0,0,0) + im.size();
		cv::Mat scratch = cv::Mat::zeros(im.size(), CV_8U);

		// the current insertion position in the vector
		size_t keep = 0;

		/* iterate through the boxes, checking:
		 * 1) has the area under the box been painted?
		 * 2) if so, is it under the threshold?
		 * 3) if so, keep this box and paint the area
		 * 4) repeat
		 */
		for (size_t n = 0; n < N; ++n) {
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
		const size_t N = candidates.size();
		mask = cv::Mat::zeros(im.size(), CV_8U);
		cv::Rect bounds = cv::Rect(0,0,0,0) + im.size();

		for (size_t n = 0; n < N; ++n) {
			cv::Rect box = candidates[n].boundingBox() & bounds;
			mask(box).setTo(n+1, mask(box) == 0);
		}
	}

};

#endif /* CANDIDATE_HPP_ */
