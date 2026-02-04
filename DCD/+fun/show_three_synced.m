function show_three_synced(img_am, img_fm, img_fuse, frame_rate, k0, play_speed)
% IO contract:
%   Inputs:
%     img_*     : uint8 [W,H,N]
%     frame_rate: positive scalar
%     k0        : initial index (1-based)
%     play_speed: scalar pacing factor

num_frames = min([size(img_am,3), size(img_fm,3), size(img_fuse,3)]);
k0 = max(1, min(num_frames, k0));

hFig = figure('Name','AM vs FM vs DCD ', 'NumberTitle','off');

ax1 = subplot(1,3,1); h1 = imshow(img_am(:,:,k0), []); title(ax1, ['AM  ', num2str(k0)]);
ax2 = subplot(1,3,2); h2 = imshow(img_fm(:,:,k0), []); title(ax2, ['FM  ', num2str(k0)]);
ax3 = subplot(1,3,3); h3 = imshow(img_fuse(:,:,k0), []); title(ax3, ['DCD  ', num2str(k0)]);

for fi = 1:num_frames
    if ~isvalid(hFig), break; end
    set(h1, 'CData', img_am(:,:,fi));
    set(h2, 'CData', img_fm(:,:,fi));
    set(h3, 'CData', img_fuse(:,:,fi));
    title(ax1, ['AM  ', num2str(fi)]);
    title(ax2, ['FM  ', num2str(fi)]);
    title(ax3, ['DCD  ', num2str(fi)]);
    drawnow;
    pause(play_speed / frame_rate);
end
end
