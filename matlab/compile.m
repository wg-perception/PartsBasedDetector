% =============
% Octave/Matlab
octave = isoctave();
if octave
    disp('Compiling for Octave...');
    gcc = 'mkoctfile --mex';
    cd oct;
else
    disp('Compiling for Matlab...');
    gcc = 'mex';
    cd mex;
end

% =============
% Learning code
disp('Learning:');
files = {'qp_one_sparse.cc', 'score.cc', 'lincomb.cc'};
octaveflags = '';
matlabflags = '-O -largeArrayDims';

for n = 1:length(files)
    if octave, cmd = [gcc ' ' octaveflags ' ' files{n}];
    else       cmd = [gcc ' ' matlabflags ' ' files{n}];
    end
    disp(['  ' cmd]); fflush(stdout);
    eval(cmd);
end

% =============
% Detection code
disp('Detection:');
files = {'resize.cc', 'reduce.cc', 'dt.cc', 'shiftdt.cc', 'features.cc'};
octaveflags = '';
matlabflags = '-O';
if isunix()
  % use one of the following depending on your setup
  % 1 is fastest, 3 is slowest 
  % 1) multithreaded convolution using blas
  %files{end+1} = '-o fconv -lmwblas fconvblas.cc';
  % 2) mulththreaded convolution without blas
  files{end+1} = '-o fconv fconvMT.cc';
  % 3) basic convolution, very compatible
  % files{end+1} = '-o fconv fconv.cc'
elseif ispc()
  files{end+1} = '-o fconv fconv.cc';
end

for n = 1:length(files)
    if octave, cmd = [gcc ' ' octaveflags ' ' files{n}];
    else       cmd = [gcc ' ' matlabflags ' ' files{n}];
    end
    disp(['  ' cmd]); fflush(stdout);
    eval(cmd);
end

% =============
% Cleanup 
if octave
    delete *.o;
end
cd ..;
