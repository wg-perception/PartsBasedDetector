% fprintflush
% calls fprintf then immediately flushes the buffer
function fprintflush(varargin)
    fprintf(varargin{:});
    fflush(stdout);
end
