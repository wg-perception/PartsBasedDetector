% modelTransfer
%   model = modelTransfer(model, format) transforms a parts based model in
%   different formats, to the format used by the PartsBasedDetector. The
%   format specifier currently supports two formats:
%
%     'VOC':
%         P. Felzenswalb, D. McAllester, D. Ramanan.
%         "Object detection with discriminatively trained part based
%         models", PAMI 2010
%         http://www.cs.brown.edu/~pff/latent/
%
%     'Face':
%         X. Zhu, D. Ramanan. "Face detection, pose estimation and landmark
%         localization in the wild", CVPR 2012
%         http://www.ics.uci.edu/~xzhu/face/
%
%   The final output model produced is compatible with the following paper:
%     Y. Yang, D. Ramanan. "Articulated pose estimation with flexible
%     mixutres of parts", CVPR 2011
%     http://phoenix.ics.uci.edu/software/pose/
%     https://github.com/hbristow/PartsBasedDetector/


%  Software License Agreement (BSD License)
%
%  Copyright (c) 2012, UC Irvine
%  All rights reserved.
%
%  Redistribution and use in source and binary forms, with or without
%  modification, are permitted provided that the following conditions
%  are met:
%
%   * Redistributions of source code must retain the above copyright
%     notice, this list of conditions and the following disclaimer.
%   * Redistributions in binary form must reproduce the above
%     copyright notice, this list of conditions and the following
%     disclaimer in the documentation and/or other materials provided
%     with the distribution.
%   * Neither the name of Willow Garage, Inc. nor the names of its
%     contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%
%  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
%  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
%  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
%  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
%  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
%  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
%  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
%  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%  POSSIBILITY OF SUCH DAMAGE.
%
%  File:    modelTransfer.m
%  Author:  Deva Ramanan, Xiangxin Zhu, Hilton Bristow 
%  Created: July 18, 2012
function model = modelTransfer(model, format)

  switch format
    case 'VOC'
      model = modelTransferVOC2Face(model);
      model = modelTransferFace2Pose(model);
    case 'Face'
      model = modelTransferFace2Pose(model);
    otherwise
      disp(['Unknown source format ' format '. Options are: Face, VOC']);
  end
end


% ---------------------------------------------------------------
% VOC -> Face
% ---------------------------------------------------------------
function model = modelTransferVOC2Face(vocmodel)

  disp('Converting VOC model to Face model...');
  m = vocmodel;
  
  if nargin < 2,
    %components = 1:length(model.rules{model.start});
    components = 1:2:length(m.rules{m.start});
  end
  layer = 1;


  nd  = 0;
  nf  = 0;
  len = 0;
  nc  = 0;
  for c = components,
    % Add offset
    offset = m.rules{m.start}(c).offset;
    x.w = offset.w;
    x.i = len + 1;
    x.anchor = [0 0 0];
    nd  = nd  + 1;
    model.defs(nd) = x;
    len = len + prod(size(x.w));
    x = [];

    rhs = m.rules{m.start}(c).rhs;  
    % assume the root filter is first on the rhs of the start rules
    if m.symbols(rhs(1)).type == 'T'
      % handle case where there's no deformation model for the root
      root = m.symbols(rhs(1)).filter;
    else
      % handle case where there is a deformation model for the root
      root = m.symbols(m.rules{rhs(1)}(layer).rhs).filter;
    end

    % Add root filter
    x.w = m.filters(root).w;
    x.i = len + 1;
    nf  = nf  + 1;
    model.filters(nf) = x;
    len = len + prod(size(x.w));

    i = 1;
    comp(i).filterid = nf;
    comp(i).defid    = nd;
    comp(i).parent   = 0;

    for i = 2:length(rhs)
      % Add part deformation
      x.w    = m.rules{rhs(i)}(layer).def.w;
      x.anchor = m.rules{m.start}(c).anchor{i} + [1 1 0];
      x.i = len + 1;
      nd  = nd  + 1;
      model.defs(nd) = x;
      len = len + prod(size(x.w));
      x = [];

      % Add part filter
      fi  = m.symbols(m.rules{rhs(i)}(layer).rhs).filter;
      x.w = m.filters(fi).w;
      x.i = len + 1;
      nf  = nf  + 1;
      model.filters(nf) = x;
      len = len + prod(size(x.w));

      % Add part to component
      comp(i).filterid = nf;
      comp(i).defid    = nd;
      comp(i).parent   = 1;
    end 
    nc = nc + 1;
    model.components{nc} = comp;
  end

  model.maxsize  = m.maxsize;
  model.minsize  = m.minsize;
  model.len      = len;
  model.interval = m.interval;
  model.sbin     = m.sbin;
  model.flip     = 1;
  model.thresh   = -0.6;
end


% ---------------------------------------------------------------
% Face -> Pose
% ---------------------------------------------------------------
function model = modelTransferFace2Pose(facemodel)

  disp('Converting Face model to Pose model...');
  m = facemodel;
  
  % copy common primitives
  model = [];
  model.interval = 10;
  model.sbin = m.sbin;
  model.maxsize = m.maxsize;
  model.thresh = m.thresh;

  % filters
  model.filters = m.filters;

  % number of mixtures
  num_component = length(m.components);
  model.components = cell(1,num_component);

  % allocate memory for biases
  % There is only one global bias for each mixture in facemodel
  % We need one "dummy" zero bias to fill the pairwise biases
  % in pose model.
  model.bias = struct('w',[]);
  model.bias(num_component+1) = model.bias;
  model.bias(num_component+1).w = 0; % "dummy" 0 for pairwise biases

  % defs
  model.defs = [];
  def_count = 0;

  % loop over mixtures
  for i = 1:num_component
      comp = facemodel.components{i};

      % bias
      % first def in facemodel is the global bias of this mixture
      b = m.defs(comp(1).defid).w; 
      assert(length(b) == 1);
      model.bias(i).w = b;

      % parts
      % allocate memory for parts
      npart = length(comp);
      parts = struct('biasid',[],'defid',[],'filterid',[],'parent',[]);
      parts(npart) = parts;
      for j = 1:npart

          if j == 1 % root 
              parts(j).defid = [];
              parts(j).biasid = i;
          else
              % add def to model.defs
              def.w = m.defs(comp(j).defid).w;
              def.anchor = m.defs(comp(j).defid).anchor;
              model.defs = [model.defs def];
              def_count = def_count+1;

              parts(j).defid = def_count;
              parts(j).biasid = num_component+1; % 0 pairwise bias
          end
          parts(j).filterid = comp(j).filterid;
          parts(j).parent = comp(j).parent;
      end
      model.components{i} = parts;   
  end
end