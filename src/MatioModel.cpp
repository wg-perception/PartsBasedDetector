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
 *  File:    MatlabIOModel.cpp
 *  Author:  Tim Sheerman-Chase
 *  Created: Apr 1, 2013
 */

#include <exception>
#include <iostream>
#include <boost/filesystem.hpp>
#include <matio.h>
#include "MatioModel.hpp"
using namespace std;

/*! @brief deserialize a Matlab .Mat file into memory
 *
 * deserialize a valid version 5 .Mat file using the underlying
 * matio parser, and populate the model fields. If any of the fields
 * do not exist, or a bad type cast is attempted, an exception will be thrown
 *
 * @param filename the path to the model file
 * @return true if the file was found, opened and verified to be a valid Matlab
 * version 5 file
 * @throws boost::bad_any_cast, exception
 */
bool MatioModel::deserialize(const std::string& filename)
{
	//Open output file for reading via matio
    mat_t *matfp = Mat_Open(filename.c_str(),MAT_ACC_RDWR);
    if ( NULL == matfp ) {
        return false;
    }

	cout << "Matrix open" << endl;

	//Read struct container
	matvar_t *cont1 = Mat_VarReadNext(matfp);
	if(cont1==NULL) return false;

	//cout << "Found " << cont1->name << endl;;	
	//cout << Mat_VarGetNumberOfFields(cont1) << endl;

	//Read struct members
	matvar_t *model = Mat_VarGetStructField(cont1, (void *)"model", BY_NAME, 0);
	if(model==NULL) {cout << "Model not found" << endl; return false;}

	matvar_t *name = Mat_VarGetStructField(cont1, (void *)"name", BY_NAME, 0);
	if(name==NULL) {cout << "Model name not found" << endl; return false;}

	matvar_t *pa = Mat_VarGetStructField(cont1, (void *)"pa", BY_NAME, 0);
	if(pa==NULL) {cout << "Model pa not found" << endl; return false;}

	matvar_t *sbin = Mat_VarGetStructField(cont1, (void *)"sbin", BY_NAME, 0);
	if(sbin==NULL) {cout << "Model sbin not found" << endl; return false;}



	return true;
}


bool MatioModel::serialize(const std::string& filename) const
{
	/* TODO: implement */
	filename[0];
	return false;
}

