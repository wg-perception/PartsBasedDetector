% --------------------
% specify model parameters
% number of mixtures for 6 parts
K = [4 4 4 4 4 4];

% Tree structure for 6 parts: pa(i) is the parent of part i
% This structure is implicity assumed during data preparation
% (PARSE_data.m) and evaluation (PARSE_eval_pcp)
pa = [0 1 1 3 2 4];

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
