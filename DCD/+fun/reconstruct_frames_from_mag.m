function image_frames = reconstruct_frames_from_mag( ...
    magnitude, width, height, samples_per_frame, sync_offset_x, sync_offset_y)
% IO contract:
%   Inputs:
%     magnitude         : real vector
%     width,height      : positive integers
%     samples_per_frame : positive integer
%     sync_offset_x/y   : integer offsets (circular)
%   Output:
%     image_frames      : uint8 array [width, height, num_frames]

L = length(magnitude);
num_frames = floor(L / samples_per_frame);

image_frames = zeros(width, height, num_frames, 'uint8');

for frame_idx = 1:num_frames
    start_idx = (frame_idx - 1) * samples_per_frame + 1;
    end_idx   = start_idx + samples_per_frame - 1;
    if end_idx > L
        break;
    end

    frame_sig = magnitude(start_idx:end_idx);

    pixel_indices = linspace(1, length(frame_sig), width * height);
    pixel_values  = interp1(1:length(frame_sig), frame_sig, pixel_indices, 'linear', 'extrap');
    pixel_values  = pixel_values(1:(width * height));

    reshaped_frame = reshape(pixel_values, [height, width]);
    shifted_frame  = circshift(reshaped_frame, [-sync_offset_y, -sync_offset_x]);

    mn = min(shifted_frame(:));
    mx = max(shifted_frame(:));
    if mx > mn
        normalized_frame = uint8(255 * (shifted_frame - mn) / (mx - mn));
    else
        normalized_frame = uint8(zeros(size(shifted_frame)));
    end

    image_frames(:, :, frame_idx) = normalized_frame';
end
end
