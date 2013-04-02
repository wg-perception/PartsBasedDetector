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
#include <assert.h>
#include "MatioModel.hpp"
using namespace std;

//************ Utility Functions ****************************

//! convert a vector of integers from Matlab 1-based indexing to C++ 0-based indexing
static inline void zeroIndex(vectori& idx) {
	for (unsigned int n = 0; n < idx.size(); ++n) idx[n] -= 1;
}

//! convert an integer from Matlab 1-based indexing to C++ 0-based indexing
static inline void zeroIndex(int& idx) {
	idx -= 1;
}

//! convert a vector of Point from Matlab 1-based indexing to C++ 0-based indexing
static inline void zeroIndex(vectorPoint& pt) {
	cv::Point one(1,1);
	for (unsigned int n = 0; n < pt.size(); ++n) pt[n] = pt[n] - one;
}

template<class mytype> mytype Matio2DToVec(matvar_t *m)
{
	assert(m->data_type==MAT_T_DOUBLE);
	assert(m->rank==2);

	double arr[m->dims[0]*m->dims[1]];
	int start[] = {0,0};
	int stride[] = {1,1};
	int *edge = m->dims;
	int ret = Mat_VarReadData(m->fp, m, arr, start, stride, edge);
	assert(ret==0);
	mytype out;
	for(unsigned int i=0;i<m->dims[0]*m->dims[1];i++)
		out.push_back(arr[i]);
	return out;
}

//************* Main Class ************************************************************

bool MatioModel::readModelData(mat_t *matfp, matvar_t *model)
{
	matvar_t *filters = Mat_VarGetStructField(model, (void *)"filters", BY_NAME, 0);
	if(filters==NULL) {cout << "Model filters not found" << endl;}

	matvar_t *components = Mat_VarGetStructField(model, (void *)"components", BY_NAME, 0);
	if(components==NULL) {cout << "Model components not found" << endl;}

	matvar_t *defs = Mat_VarGetStructField(model, (void *)"defs", BY_NAME, 0);
	if(defs==NULL) {cout << "Model filters not found" << endl;}

	matvar_t *bias = Mat_VarGetStructField(model, (void *)"bias", BY_NAME, 0);
	if(bias==NULL) {cout << "Model bias not found" << endl;}

	//Return if something is missing
	if(filters==NULL || components==NULL || defs==NULL || bias == NULL)
	{
		if(filters!=NULL) Mat_VarFree(filters);
		if(components!=NULL) Mat_VarFree(components);
		if(defs!=NULL) Mat_VarFree(defs);
		if(bias!=NULL) Mat_VarFree(bias);
		return false;
	}

	assert(filters->data_type==MAT_T_STRUCT);
	//cout << filters->rank << "," << filters->dims[0] <<"," << filters->dims[1] << endl;

	for(int structInd = 0; structInd < filters->dims[1]; structInd++)
	{
		//Get filteri values
		//matvar_t *filtersi = Mat_VarGetStructField(filters, (void *)"i", BY_NAME, structInd);
		//if(filtersi==NULL) {cout << "Model filtersi not found" << endl;}	
		//assert(filtersi->data_type==MAT_T_DOUBLE);

		//Get filterw values
		matvar_t *filtersw = Mat_VarGetStructField(filters, (void *)"w", BY_NAME, structInd);
		if(filtersw==NULL) {cout << "Model filtersw not found" << endl;}	
		assert(filtersw->data_type==MAT_T_DOUBLE);
		assert(filtersw->rank==3);

		//cout <<structInd << ","<<filtersw->data_type<< "," << filtersw->dims[0] <<"," << filtersw->dims[1]<<"," << filtersw->dims[2] << endl;
		
		int C = filtersw->dims[2];
		this->flen_ = C;
		cv::Mat filter_flat(cv::Size(C, 1), cv::DataType<double>::type);

		double *buff = new double[filtersw->dims[0]*filtersw->dims[1]*filtersw->dims[2]];
		int start[] = {0,0,0};
		int stride[] = {1,1,1};
		int edge[] = {filtersw->dims[0],filtersw->dims[1],filtersw->dims[2]};
		int ret = Mat_VarReadData(matfp, filtersw, buff, start, stride, edge);
		assert(ret==0);
		for(int c=0; c<C; c++)
			filter_flat.at<double>(c,1) = buff[c];
		delete buff;

		this->filtersw_.push_back(filter_flat);
	}

	//Copy components into memory
	assert(components->data_type==MAT_T_CELL);
	//cout <<components->data_type<< "," << components->dims[0] <<"," << components->dims[1]<<"," << components->dims[2] << endl;
	biasid_.resize(components->dims[1]);
	filterid_.resize(components->dims[1]);
	defid_.resize(components->dims[1]);
	parentid_.resize(components->dims[1]);

	for(int cellInd = 0; cellInd < components->dims[1]; cellInd++)
	{
		matvar_t *cell = Mat_VarGetCell(components, cellInd);
		assert(cell!=NULL);
		assert(cell->data_type==MAT_T_STRUCT);
		biasid_[cellInd].resize(cell->dims[1]);
		filterid_[cellInd].resize(cell->dims[1]);
		defid_[cellInd].resize(cell->dims[1]);
		parentid_[cellInd].resize(cell->dims[1]);

		for(int structInd = 0; structInd < cell->dims[1]; structInd++)
		{
			matvar_t *defid = Mat_VarGetStructField(cell, (void *)"defid", BY_NAME, structInd);
			matvar_t *filterid = Mat_VarGetStructField(cell, (void *)"filterid", BY_NAME, structInd);
			matvar_t *parent = Mat_VarGetStructField(cell, (void *)"parent", BY_NAME, structInd);
			matvar_t *biasid = Mat_VarGetStructField(cell, (void *)"biasid", BY_NAME, structInd);
			assert(defid!=NULL && filterid != NULL && parent != NULL && biasid != NULL);

			this->biasid_[cellInd][structInd] = Matio2DToVec<vectori>(biasid);
			this->parentid_[cellInd][structInd] = *(double *)parent->data;
			this->filterid_[cellInd][structInd] = Matio2DToVec<vectori>(filterid);
			this->defid_[cellInd][structInd]    = Matio2DToVec<vectori>(biasid);

			//re-index from zero (Matlab uses 1-based indexing)
			zeroIndex(biasid_[cellInd][structInd]);
			zeroIndex(parentid_[cellInd][structInd]);
			zeroIndex(filterid_[cellInd][structInd]);
			zeroIndex(defid_[cellInd][structInd]);
		}
	}

	//Copy defs to memory
	for(int structInd = 0; structInd < defs->dims[1]; structInd++)
	{
		matvar_t *defsw = Mat_VarGetStructField(defs, (void *)"w", BY_NAME, structInd);
		matvar_t *defsanchor = Mat_VarGetStructField(defs, (void *)"anchor", BY_NAME, structInd);
		assert(defsw!=NULL && defsanchor!=NULL);

		vector<float> wv = Matio2DToVec<vectorf>(defsw);
		this->defw_.push_back(wv);

		assert(defsanchor->data_type==MAT_T_DOUBLE);
		double p[defsanchor->dims[0]*defsanchor->dims[1]];
		int start2[] = {0,0};
		int stride2[] = {1,1};
		int *edge2 = defsanchor->dims;
		int ret = Mat_VarReadData(matfp, defsanchor, p, start2, stride2, edge2);
		assert(ret==0);
		this->anchors_.push_back(cv::Point(p[0], p[1]));
	}
	zeroIndex(this->anchors_);

	//Read bias into memory
	for(int structInd = 0; structInd < bias->dims[1]; structInd++)
	{
		matvar_t *biasw = Mat_VarGetStructField(bias, (void *)"w", BY_NAME, structInd);
		assert(biasw!=NULL);
		assert(biasw->data_type==MAT_T_DOUBLE);
		this->biasw_.push_back(*(double *)biasw->data);
	}

	//if(filters!=NULL) Mat_VarFree(filters);
	//if(components!=NULL) Mat_VarFree(components);
	//if(defs!=NULL) Mat_VarFree(defs);
	//if(bias!=NULL) Mat_VarFree(bias);
	return true;
}

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

	//Read struct container
	matvar_t *cont1 = Mat_VarReadNext(matfp);
	if(cont1==NULL) {Mat_Close(matfp); return false;}

	//Read struct members
	matvar_t *model = Mat_VarGetStructField(cont1, (void *)"model", BY_NAME, 0);
	if(model==NULL) {cout << "Model data not found" << endl;}

	matvar_t *name = Mat_VarGetStructField(cont1, (void *)"name", BY_NAME, 0);
	if(name==NULL) {cout << "Model name not found" << endl;}

	//matvar_t *pa = Mat_VarGetStructField(cont1, (void *)"pa", BY_NAME, 0);
	//if(pa==NULL) {cout << "Model pa not found" << endl;}

	matvar_t *sbin = Mat_VarGetStructField(cont1, (void *)"sbin", BY_NAME, 0);
	if(sbin==NULL) {cout << "Model sbin not found" << endl;}

	//Return if something is missing
	if(model==NULL || name==NULL || sbin==NULL) //|| pa==NULL
	{
		if(model!=NULL) Mat_VarFree(model);
		if(name!=NULL) Mat_VarFree(name);
		//if(pa!=NULL) Mat_VarFree(pa);
		if(sbin!=NULL) Mat_VarFree(sbin);
		Mat_VarFree(cont1);
	}
	
	//cout << (name->data_type == MAT_T_UINT8) << "," << name->data_size << endl;
	//cout << (model->data_type == MAT_T_STRUCT) << "," << model->data_size << endl;
	this->name_ = (char *)name->data;

	//assert(pa->data_type==MAT_T_DOUBLE);
	//assert(sizeof(double) == pa->data_size);
	//double *paVal = (double *)pa->data;

	assert(sbin->data_type==MAT_T_DOUBLE);
	assert(sizeof(double) == sbin->data_size);
	double *sbinVal = (double *)sbin->data;
	this->binsize_ = *sbinVal;

	this->norient_ = 18;

	bool ret = this->readModelData(matfp, model);

	//Clear matio objects
	//if(model!=NULL) Mat_VarFree(model);
	//if(name!=NULL) Mat_VarFree(name);
	//if(pa!=NULL) Mat_VarFree(pa);
	//if(sbin!=NULL) Mat_VarFree(sbin);
	Mat_VarFree(cont1);
	Mat_Close(matfp);
	return ret;
}


bool MatioModel::serialize(const std::string& filename) const
{
	/* TODO: implement */
	filename[0];
	return false;
}


