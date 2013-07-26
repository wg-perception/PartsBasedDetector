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
 *  File:    DynamicProgram.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#include "Math.hpp"
#include "DynamicProgram.hpp"
using namespace cv;
using namespace std;


/*! @brief Get the min of a dynamic program
 *
 * Get the min of a dynamic program by starting at the leaf nodes,
 * and passing the scores up to the root node. This is essentially
 * a tail recursive algorithm which we can unfold since we know that
 * the parts are sorted from the root to the leaves
 *
 * The algorithm involves 3 steps:
 * 		(1) Apply distance transform
 * 		(2) Shift by the anchor position of the part wrt the parent
 * 		(3) Downsample if necessary
 *
 * @param parts the parts tree, referenced by the root
 * @param scores the probability densities (pdfs) of part locations (fine to coarse)
 * @param Ix the detection indices in the x direction
 * @param Iy the detection indices in the y direction
 * @param Ik the best mixture at each pixel
 * @param rootv the root scores, across scale
 * @param rooti the root indices, across scale
 *
 */
template<typename T>
void DynamicProgram<T>::min(Parts& parts, vector2DMat& scores, vector4DMat& Ix, vector4DMat& Iy, vector4DMat& Ik, vector2DMat& rootv, vector2DMat& rooti) {

	// initialize the outputs, preallocate vectors to make them thread safe
	// TODO: better initialisation of Ix, Iy, Ik
	const size_t nscales = scores.size();
	const size_t ncomponents = parts.ncomponents();
	Ix.resize(nscales, vector3DMat(ncomponents));
	Iy.resize(nscales, vector3DMat(ncomponents));
	Ik.resize(nscales, vector3DMat(ncomponents));
	rootv.resize(nscales, vectorMat(ncomponents));
	rooti.resize(nscales, vectorMat(ncomponents));

	// for each scale, and each component, update the scores through message passing
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (size_t nc = 0; nc < nscales*ncomponents; ++nc) {

		// calculate the inner loop variables from the dual variables
		const size_t n = floor(nc / ncomponents);
		const size_t c = nc % ncomponents;

		// allocate the inner loop variables
		Ix[n][c].resize(parts.nparts(c));
		Iy[n][c].resize(parts.nparts(c));
		Ik[n][c].resize(parts.nparts(c));
		vectorMat ncscores(scores[n].size());

		for (int p = parts.nparts(c)-1; p > 0; --p) {

			// get the component part (which may have multiple mixtures associated with it)
			ComponentPart cpart = parts.component(c, p);
			const size_t nmixtures  = cpart.nmixtures();
			const size_t pnmixtures = cpart.parent().nmixtures();
			Ix[n][c][p].resize(pnmixtures);
			Iy[n][c][p].resize(pnmixtures);
			Ik[n][c][p].resize(pnmixtures);

			// intermediate results for mixtures of this part
			vectorMat scoresp;
			vectorMat Ixp;
			vectorMat Iyp;

			for (size_t m = 0; m < nmixtures; ++m) {

				// raw score outputs
				Mat_<T> score_in, score_dt;
				Mat_<int> Ix_dt, Iy_dt;
				if (cpart.score(ncscores, m).empty()) {
					score_in = cpart.score(scores[n], m);
				} else {
					score_in = cpart.score(ncscores, m);
				}

				// get the anchor position
				Point anchor = cpart.anchor(m);

				// compute the distance transform
				vectorf w = cpart.defw(m);
				Quadratic fx(-w[0], -w[1]);
				Quadratic fy(-w[2], -w[3]);
				dt_.compute(score_in, fx, fy, anchor, score_dt, Ix_dt, Iy_dt);
				scoresp.push_back(score_dt);
				Ixp.push_back(Ix_dt);
				Iyp.push_back(Iy_dt);
			}

			for (size_t m = 0; m < pnmixtures; ++m) {
				vectorMat weighted;
				// weight each of the child scores
				// TODO: More elegant way of handling bias
				for (size_t mm = 0; mm < nmixtures; ++mm) {
					weighted.push_back(scoresp[mm] + cpart.bias(mm)[m]);
				}
				// compute the max over the mixtures
				Mat maxv, maxi;
				Math::reduceMax<T>(weighted, maxv, maxi);

				// choose the best indices
				Mat Ixm, Iym;
				Math::reducePickIndex<int>(Ixp, maxi, Ixm);
				Math::reducePickIndex<int>(Iyp, maxi, Iym);
				Ix[n][c][p][m] = Ixm;
				Iy[n][c][p][m] = Iym;
				Ik[n][c][p][m] = maxi;

				// update the parent's score
				ComponentPart parent = cpart.parent();
				if (parent.score(ncscores,m).empty()) parent.score(scores[n],m).copyTo(parent.score(ncscores,m));
				parent.score(ncscores,m) += maxv;
				if (parent.self() == 0) {
					ComponentPart root = parts.component(c);
				}
			}
		}
		// add bias to the root score and find the best mixture
		ComponentPart root = parts.component(c);
		Mat rncscore = root.score(ncscores,0);
		T bias = root.bias(0)[0];
		vectorMat weighted;
		// weight each of the child scores
		for (size_t m = 0; m < root.nmixtures(); ++m) {
			weighted.push_back(root.score(ncscores,m) + bias);
		}
		Math::reduceMax<T>(weighted, rootv[n][c], rooti[n][c]);
	}
}


/*! @brief get the argmin of a dynamic program
 *
 * Get the minimum argument of a dynamic program by traversing down the tree of
 * a dynamic program, returning the locations of the best nodes
 * @param parts the tree of parts, referenced by the root
 * @param rootv the root scores, across scale
 * @param rooti the root indices, across scale
 * @param scales the scales (used to calculate bounding box size)
 * @param Ix the detection indices in the x direction
 * @param Iy the detection indices in the y direction
 * @param Ik the best mixture at each pixel
 * @param candidates
 */
template<typename T>
void DynamicProgram<T>::argmin(Parts& parts, const vector2DMat& rootv, const vector2DMat& rooti, const vectorf scales, const vector4DMat& Ix, const vector4DMat& Iy, const vector4DMat& Ik, vectorCandidate& candidates) {

	// for each scale, and each component, traverse back down the tree to retrieve the part positions
	const size_t nscales = scales.size();
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (size_t n = 0; n < nscales; ++n) {
		T scale = scales[n];
		for (size_t c = 0; c < parts.ncomponents(); ++c) {

			// get the scores and indices for this tree of parts
			const vector2DMat& Iknc = Ik[n][c];
			const vector2DMat& Ixnc = Ix[n][c];
			const vector2DMat& Iync = Iy[n][c];
			const size_t nparts = parts.nparts(c);

			// threshold the root score
			Mat over_thresh = rootv[n][c] > thresh_;
			Mat rootmix     = rooti[n][c];
			vectorPoint inds;
			Math::find(over_thresh, inds);

			for (size_t i = 0; i < inds.size(); ++i) {
				Candidate candidate;
				candidate.setComponent(c);
				vectori     xv(nparts);
				vectori     yv(nparts);
				vectori     mv(nparts);
				for (size_t p = 0; p < nparts; ++p) {
					ComponentPart part = parts.component(c, p);
					// calculate the child's points from the parent's points
					size_t x, y, m;
					if (part.isRoot()) {
						x = xv[0] = inds[i].x;
						y = yv[0] = inds[i].y;
						m = mv[0] = rootmix.at<int>(inds[i]);
					} else {
						int idx = part.parent().self();
						x = xv[idx];
						y = yv[idx];
						m = mv[idx];
						xv[p] = Ixnc[p][m].at<int>(y,x);
						yv[p] = Iync[p][m].at<int>(y,x);
						mv[p] = Iknc[p][m].at<int>(y,x);
					}

					// calculate the bounding rectangle and add it to the Candidate
					Point pone = Point(1,1);
					Point xy1 = (Point(xv[p],yv[p])-pone)*scale;
					Point xy2 = xy1 + Point(part.xsize(mv[p]), part.ysize(mv[p]))*scale - pone;
					if (part.isRoot()) 
					  candidate.addPart(Rect(xy1, xy2), rootv[n][c].at<T>(inds[i]));
					else
					  candidate.addPart(Rect(xy1, xy2), 0.0);
				}
				#ifdef _OPENMP
				#pragma omp critical(addcandidate)
				#endif
				{
					candidates.push_back(candidate);
				}
			}
		}
	}
}



// declare all specializations of the template (this must be the last declaration in the file)
template class DynamicProgram<float>;
template class DynamicProgram<double>;

