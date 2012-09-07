% isoctave()
% determine whether the running interpreter is
% an instance of Matlab or Octave
function val = isoctave()
    val = exist('OCTAVE_VERSION', 'builtin');
end
