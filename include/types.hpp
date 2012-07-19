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
 *  File:    types.hpp
 *  Author:  Hilton Bristow
 *  Created: Jul 8, 2012
 */

#ifndef TYPES_HPP_
#define TYPES_HPP_
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class MatlabIOContainer;
class Candidate;

// common typedefs
// 1D
typedef std::vector<int>            vectori;
typedef std::vector<float>			vectorf;
typedef std::vector<cv::Mat>        vectorMat;
typedef std::vector<cv::Point>      vectorPoint;
typedef std::vector<cv::Point3i>    vectorPoint3;
typedef std::vector<Candidate>      vectorCandidate;
typedef std::vector<MatlabIOContainer> vectorMatlabIOContainer;
typedef std::vector<cv::Ptr<cv::FilterEngine> > vectorFilterEngine;
// 2D
typedef std::vector<vectori>     	vector2Di;
typedef std::vector<vectorf>   		vector2Df;
typedef std::vector<vectorMat> 		vector2DMat;
typedef std::vector<vectorMatlabIOContainer> vector2DMatlabIOContainer;
typedef std::vector<std::vector<cv::Ptr<cv::FilterEngine> > > vector2DFilterEngine;
// 3D
typedef std::vector<vector2Di>      vector3Di;
typedef std::vector<vector2DMat>    vector3DMat;
// 4D
typedef std::vector<vector3DMat>    vector4DMat;



#endif /* TYPES_HPP_ */
