function magnitude_am = demod_am(sig_I, sig_Q)
% IO contract:
%   Input : sig_I, sig_Q
%   Output: magnitude_am

magnitude_am = sqrt(sig_I.^2 + sig_Q.^2);
end
