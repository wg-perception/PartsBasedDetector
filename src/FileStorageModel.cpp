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
 *  File:    FileStorageModel.cpp
 *  Author:  Hilton Bristow
 *  Created: Jul 27, 2012
 */

#include <sstream>
#include "FileStorageModel.hpp"

bool FileStorageModel::serialize(const std::string& filename) const {

	// open the storage container for writing
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::WRITE);

	// write the primitives
	fs << "name" 	 << name_;
	fs << "interval" << nscales_;
	fs << "thresh" 	 << thresh_;
	fs << "sbin" 	 << binsize_;
	fs << "norient"  << norient_;
	fs << "flen"     << flen_;

	// write the single depth vectors
	fs << "filtersw" << filtersw_;
	fs << "biasw"    << biasw_;
	fs << "anchors"  << anchors_;

	// write the 2D deformation weights
	fs << "defs" << "[";
	for (size_t n = 0; n < defw_.size(); ++n) {
		fs << defw_[n];
	}
	fs << "]";

	// write the deeper indexing vectors
	fs << "indexers" << "{";
	const size_t ncomponents = filterid_.size();
	for (size_t c = 0; c < ncomponents; ++c) {
		std::ostringstream cstr;
		cstr << "component-" << c;
		fs << cstr.str() << "{";
		const size_t nparts = filterid_[c].size();
		for (size_t p = 0; p < nparts; ++p) {
			std::ostringstream pstr;
			pstr << "part-" << p;
			fs << pstr.str() << "{";
			fs << "parentid" << parentid_[c][p];
			fs << "filterid" << filterid_[c][p];
			fs << "biasid"   << biasid_[c][p];
			fs << "defid"    << defid_[c][p];
			fs << "}";
		}
		fs << "}";
	}
	fs << "}";


	// close the file store
	fs.release();
	return true;
}

bool FileStorageModel::deserialize(const std::string& filename) {

	// open the storage container for writing
	cv::FileStorage fs;
	bool ok = fs.open(filename, cv::FileStorage::READ);
	if (!ok) return false;

	// read the primitives
	fs["name"] 	   >> name_;
	fs["interval"] >> nscales_;
	fs["thresh"]   >> thresh_;
	fs["sbin"] 	   >> binsize_;
	fs["norient"]  >> norient_;
	fs["flen"]     >> flen_;

	// read the single depth vectors
	fs["filtersw"] >> filtersw_;
	fs["biasw"]    >> biasw_;
	fs["anchors"]  >> anchors_;

	// read the 2D deformation weights
	cv::FileNode defs = fs["defs"];
	const size_t ndefs = defs.size();
	defw_.resize(ndefs);
	for (size_t n = 0; n < ndefs; ++n) {
		defs[n] >> defw_[n];
	}

	// read the indexing vectors
	cv::FileNode components = fs["indexers"];
	const size_t ncomponents = components.size();
	parentid_.resize(ncomponents);
	filterid_.resize(ncomponents);
	biasid_.resize(ncomponents);
	defid_.resize(ncomponents);
	for (size_t c = 0; c < ncomponents; ++c) {
		std::ostringstream cstr;
		cstr << "component-" << c;
		cv::FileNode parts = components[cstr.str()];
		const size_t nparts = parts.size();
		parentid_[c].resize(nparts);
		filterid_[c].resize(nparts);
		biasid_[c].resize(nparts);
		defid_[c].resize(nparts);
		for (size_t p = 0; p < nparts; ++p) {
			std::ostringstream pstr;
			pstr << "part-" << p;
			cv::FileNode part = parts[pstr.str()];
			part["parentid"] >> parentid_[c][p];
			part["filterid"] >> filterid_[c][p];
			part["biasid"]   >> biasid_[c][p];
			part["defid"]    >> defid_[c][p];
		}
	}

	// close the file store
	fs.release();
	return true;
}
