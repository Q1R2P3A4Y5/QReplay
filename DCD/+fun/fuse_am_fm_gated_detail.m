function image_frames_fuse2 = fuse_am_fm_gated_detail(image_frames_am, image_frames_fm, p)
% IO contract:
%   Inputs:
%     image_frames_am : uint8 [W,H,N]
%     image_frames_fm : uint8 [W,H,N]
%     p : struct with fields {sigma_blur,lambda,gamma,clipA,clipF}
%   Output:
%     image_frames_fuse2 : uint8 [W,H,N]
% Dependencies:
%   Image Processing Toolbox (adapthisteq, imgaussfilt, imgradientxy)

num_frames = min(size(image_frames_am,3), size(image_frames_fm,3));
image_frames_fuse2 = zeros(size(image_frames_am,1), size(image_frames_am,2), num_frames, 'uint8');

for fi = 1:num_frames
    Aeq = adapthisteq(image_frames_am(:,:,fi), 'ClipLimit', p.clipA);
    Feq = adapthisteq(image_frames_fm(:,:,fi), 'ClipLimit', p.clipF);

    Aeq = double(Aeq);
    Feq = double(Feq);

    F_low    = imgaussfilt(Feq, p.sigma_blur);
    F_detail = Feq - F_low;

    [gx,gy] = imgradientxy(Feq, 'sobel');
    G = hypot(gx, gy);
    G = G / (max(G(:)) + eps);
    W = G .^ p.gamma;

    fuse = Aeq + p.lambda * (W .* F_detail);

    fuse = fuse - min(fuse(:));
    mx = max(fuse(:));
    if mx > 0
        fuse = 255 * fuse / mx;
    end

    image_frames_fuse2(:,:,fi) = uint8(fuse);
end
end
