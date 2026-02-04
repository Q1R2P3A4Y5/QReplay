function [sig_I, sig_Q] = read_iq_float_interleaved(filename)
% IO contract:
%   Input : filename (char/string), binary float32 interleaved IQ.
%   Output: sig_I, sig_Q (double column vectors), NaN -> 0.
% Errors:
%   Throws if file cannot be opened.

f = fopen(filename, 'rb');
if f < 0
    error('Cannot open file: %s', filename);
end

values = fread(f, Inf, 'float');
fclose(f);

sig_I = values(1:2:end);
sig_Q = values(2:2:end);

sig_I(isnan(sig_I)) = 0;
sig_Q(isnan(sig_Q)) = 0;
end
