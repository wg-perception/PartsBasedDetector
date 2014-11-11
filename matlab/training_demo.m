% --------------------
% specify model parameters
% number of mixtures for 6 parts
% See "Structure" section in http://www.ics.uci.edu/~dramanan/papers/pose2011.pdf, page 5, for scholarly reference.
K = [4 4 4 4 4 4];

% Tree structure for 6 parts: pa(i) is the parent of part i
% This structure must correspond to the part labels passed to annotateParts().
% See "Deriving part type from position" section in http://www.ics.uci.edu/~dramanan/papers/pose2011.pdf, page 4, for scholarly reference.

% (PARSE_data.m) and evaluation (PARSE_eval_pcp)
pa = [0 1 1 3 2 4];

% The above example array corresponds to the tree structure below, which
% indicates that parts 0 and 3 are not attached to any other parts,
% but part 1 is a member of a chain of parts, being a attached to part 2,
% which is attached to part 4, which is attached to part 5. Here are three
% examples of what parts 1, 2, 4, and 5 could look like, given this tree:

%   1     1 2 4 5     1
%   2                 2 4
%   4                   5
%   5

% You could think of the example tree structure representing an upper arm,
% lower arm, hand, and fingers.

% Spatial resolution of HOG cell, interms of pixel width and hieght
sbin = 8;

% --------------------
% Define training and testing data
globals;
name = 'Demo model';
[pos test] = getPositiveData('/path/to/positive/data','im_regex','label_regex',0.7);
neg        = getNegativeData('/path/to/positive/data', 'im_regex');
pos        = pointtobox(pos,pa);

% --------------------
% training
model = trainmodel(name,pos,neg,K,pa,sbin);
save('Demo_model.mat', 'model', 'pa', 'sbin', 'name');
