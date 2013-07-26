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
 *  File:    Rect3.hpp
 *  Author:  Hilton Bristow
 *  Created: Aug 10, 2012
 */

#ifndef RECT3_HPP_
#define RECT3_HPP_
#include <opencv2/core/core.hpp>
#include <vector>

/*! @class Rect3_
 *  @brief 3D rectangle class
 *  This class implements a 3D rectangle which provides similar functionality
 *  to the OpenCV Rect_ class, but in 3-space
 */
template<typename T>
class Rect3_ {
public:
	// members
	T x, y, z, height, width, depth;
	//! default constructor
	Rect3_() : x(0), y(0), z(0), height(0), width(0), depth(0) {}
	//! constructor with primitive arguments
	Rect3_(T _x, T _y, T _z, T _height, T _width, T _depth) :
		x(_x), y(_y), z(_z), height(_height), width(_width), depth(_depth) {}
	//! copy constructor
	Rect3_(const Rect3_<T>& r) : x(r.x), y(r.y), z(r.z), height(r.height), width(r.width), depth(r.depth) {}
	//! constructor using points
	Rect3_(const cv::Point3_<T> pt1, const cv::Point3_<T> pt2) :
		x(pt1.x), y(pt1.y), z(pt1.z),
		height(pt2.y-pt1.y), width(pt2.x-pt1.x), depth(pt2.z-pt1.z) {}

	//! equals operator
	//Rect3_<T>& operator = (const Rect3_<T>& r) { return Rect3_<T>(r); }
	//! addition operator
	Rect3_<T>& operator + (const cv::Point3_<T>& r) { return Rect3_<T>(this->tl()+r, this->br()+r); }
	//! subtraction operator
	Rect3_<T>& operator - (const cv::Point3_<T>& r) { return Rect3_<T>(this->tl()-r, this->br()-r); }
	//! addition in-place
	Rect3_<T>& operator += (const cv::Point3_<T>& r) { x+=r.x; y+=r.y; z+=r.z; return *this; }
	//! subtraction in-place
	Rect3_<T>& operator -= (const cv::Point3_<T>& r) { x-=r.x; y-=r.y; z-=r.z; return *this; }
	//! intersection operator
	Rect3_<T>& operator & (const Rect3_<T> r) { return intersection(this, r); }
	//! convex hull operator
	Rect3_<T>& operator | (const Rect3_<T> r) { return convexHull(this, r); }
	//! stream insertion operator
	friend std::ostream& operator << (std::ostream& stream, const Rect3_<T>& r) {
		stream << "[" << r.x << ", " << r.y << ", " << r.z << " | " << r.width << ", " << r.height << ", " << r.depth << "]";
		return stream;
	}

	//! down-conversion to a Rect
	operator cv::Rect_<T> () {
		cv::Rect rect;
		rect.x = x; rect.y = y; rect.height = height; rect.width = width;
		return rect;
	}

	//! the top left corner in 3-space
	cv::Point3_<T> tl() const { return cv::Point3_<T>(x, y, z); }
	//! the bottom right corner in 3-space
	cv::Point3_<T> br() const { return cv::Point3_<T>(x+width, y+height, z+depth); }

	//! the volume of the encapsulated area
	T volume() const { return width*height*depth; }

	//! check whether the prism contains the 3D-point
	bool contains(const cv::Point3_<T>& pt) const {
		return (pt.x >= x && pt.x <= x+width &&
				pt.y >= y && pt.y <= y+height &&
				pt.z >= z && pt.z <= z+depth);
	}

	//! get the centroid of the prism
	cv::Point3_<T> centroid() const { return (this->tl() + this->br()) * 0.5; }

	/*! @brief convex hull of two rectangles
	 * The rectangular convex hull is defined as the minimum bounding rectangle
	 * that completely encapsulates both rectangles (in 3-space)
	 *
	 * @param r1 the first rectangular prism
	 * @param r2 the second rectangular prism
	 * @return the rectangular prism convex hull
	 */
	static Rect3_<T> convexHull(const Rect3_<T>& r1, const Rect3_<T>& r2) {
		return Rect3_<T>(cv::Point3_<T>(std::min(r1.tl().x, r2.tl().x),
										std::min(r1.tl().y, r2.tl().y),
										std::min(r1.tl().z, r2.tl().z)),
						 cv::Point3_<T>(std::max(r1.br().x, r2.br().x),
								 	 	std::max(r1.br().y, r2.br().y),
								 	 	std::max(r1.br().z, r2.br().z)));
	}

	/*! @brief convex hull of two rectangles
	 * The rectangular convex hull is defined as the minimum bounding rectangle
	 * that completely encapsulates both rectangles (in 3-space)
	 *
	 * @param r a std::vector of rectangular prisms
	 * @return the rectangular prism convex hull
	 */
	static Rect3_<T> convexHull(const std::vector<Rect3_<T> >& r) {
		Rect3_<T> hull = r[0];
		const int N = r.size();
		for (size_t n = 1; n < N; ++n) {
			hull = convexHull(hull, r[n]);
		}
		return hull;
	}

	/*! @brief find the intersection of two rectangular prisms in 3-space
	 *
	 * The intersection is the largest volume occupied by both rectangles.
	 * If the rectangles do not overlap, the output Rect3_ is set to
	 * all zeros
	 *
	 * @param r1 the first rectangular prism
	 * @param r2 the second rectangular prism
	 * @return a rectangular prism representing the intersection of r1 & r2
	 */
	static Rect3_<T> intersection(const Rect3_<T>& r1, const Rect3_<T>& r2) {
		Rect3_<T> inter(cv::Point3_<T>( std::max(r1.tl().x, r2.tl().x),
										std::max(r1.tl().y, r2.tl().y),
										std::max(r1.tl().z, r2.tl().z)),
						 cv::Point3_<T>(std::min(r1.br().x, r2.br().x),
								 	 	std::min(r1.br().y, r2.br().y),
								 	 	std::min(r1.br().z, r2.br().z)));
		if (inter.height < 0 || inter.width < 0 || inter.depth < 0) inter = Rect3_<T>();
		return inter;
	}

	//! destructor
	virtual ~Rect3_() {}
};

//! convenient typedef for integer type rectangular prisms
typedef Rect3_<int>    Rect3;
typedef Rect3_<float>  Rect3f;
typedef Rect3_<double> Rect3d;

#endif /* RECT3_HPP_ */
