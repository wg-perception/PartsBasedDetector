function pts_out = map_rotate_points(pts_in,im_ori,ang,flag)
% pts: N*2 points matrix. Each row is [xi yi]
% im: the original image
% ang: rotation in degree
% if flag = 'ori2new', map points in the original image to the rotated image
% if flag = 'new2ori', map points in the rotated image to the original image

switch flag
    case 'ori2new'
        direction = 0;
    case 'new2ori'
        direction = 1;
    otherwise
        error('flag has to be a string of ori2new or new2ori');
end

% initialize output
pts_out = nan(size(pts_in));

% Convert to radians
phi = ang*pi/180; 
rotate = maketform('affine', [ cos(phi)  sin(phi)  0; ...
    -sin(phi)  cos(phi)  0; ...
    0       0       1 ]);

% size of images
so = size(im_ori);
twod_size = so(1:2);

% Coordinates from center of A
hiA = (twod_size-1)/2;
loA = -hiA;

hiB = ceil(max(abs(tformfwd([loA(1) hiA(2); hiA(1) hiA(2)],rotate)))/2)*2;
loB = -hiB;

for i = 1:size(pts_in,1)
    p = [pts_in(i,2) pts_in(i,1)];
    if direction == 0
        p_new = tformfwd(p+loA,rotate)-loB;
    else
        p_new = tforminv(p+loB,rotate)-loA;
    end
    pts_out(i,2) = p_new(1);
    pts_out(i,1) = p_new(2);
end
