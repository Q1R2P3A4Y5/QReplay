function magnitude_fm = demod_fm_diffphase(sig_I, sig_Q)
% IO contract:
%   Input : sig_I, sig_Q
%   Output: magnitude_fm

x = complex(sig_I, sig_Q);
prod = x(2:end) .* conj(x(1:end-1));
magnitude_fm = angle(prod);

magnitude_fm = [magnitude_fm; mean(magnitude_fm)];
magnitude_fm = magnitude_fm - mean(magnitude_fm);
end
