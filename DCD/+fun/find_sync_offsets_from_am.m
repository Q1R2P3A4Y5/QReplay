function [sync_offset_x, sync_offset_y] = find_sync_offsets_from_am( ...
    magnitude_am, width, height, manual_adjust_hsync, manual_adjust_vsync)
% IO contract:
%   Inputs:
%     magnitude_am : real vector
%     width,height : positive integers (frame geometry)
%     manual_adjust_hsync/manual_adjust_vsync : integer offsets
%   Outputs:
%     sync_offset_x in [0,width-1], sync_offset_y in [0,height-1]

width_collapse_buffer  = zeros(1, width);
height_collapse_buffer = zeros(1, height);

L = length(magnitude_am);
for i = 1:L
    pixel_index = mod(i - 1, width * height) + 1;
    row = ceil(pixel_index / width);
    col = mod(pixel_index - 1, width) + 1;

    width_collapse_buffer(col)  = width_collapse_buffer(col)  + magnitude_am(i);
    height_collapse_buffer(row) = height_collapse_buffer(row) + magnitude_am(i);
end

conv_window_x = round(width * 0.05);
[~, best_fit_x] = max(conv(width_collapse_buffer, ones(1, conv_window_x), 'same'));

conv_window_y = round(height * 0.02);
[~, best_fit_y] = max(conv(height_collapse_buffer, ones(1, conv_window_y), 'same'));

sync_offset_x = mod(best_fit_x - 1 + manual_adjust_hsync, width);
sync_offset_y = mod(best_fit_y - 1 + manual_adjust_vsync, height);
end
