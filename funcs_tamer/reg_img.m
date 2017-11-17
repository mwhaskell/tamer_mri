function [ rerr, xmot] = reg_img (poff, xsrc, xtarg)

%% 

[nlin, ncol] = size(xsrc);

%% create phase ramps
lfreq = repmat(exp(1i * 2 * pi * [0:nlin-1] / nlin * poff(1)).', 1, ncol);
cfreq = repmat(exp(1i * 2 * pi * [0:ncol-1] / ncol * poff(2)), nlin, 1);

ksrc = ifftshift(fft2(fftshift(xsrc)));

ksrc = ksrc .* lfreq;
ksrc = ksrc .* cfreq;

xmot = fftshift(ifft2(ifftshift(ksrc)));

rerr = norm(xmot(:) - xtarg(:)) / norm(xtarg(:));










