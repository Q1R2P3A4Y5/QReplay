function main_recon_demo()

clc; close all;

cfg.filename      = 'Demo';
cfg.width         = 526;   %Can be calculated through autocorrelation function
cfg.height        = 700;
cfg.frame_rate    = 63.23; %Can be calculated through autocorrelation function
cfg.sampling_rate = 10e6;

cfg.skip          = 200;
cfg.skip_samples  = cfg.width * cfg.skip;

cfg.manual_hsync  = 0;
cfg.manual_vsync  = 0;

fuse.sigma_blur   = 2.0;
fuse.lambda       = 0.8;
fuse.gamma        = 1.0;
fuse.clipA        = 0.015;
fuse.clipF        = 0.008;

play.k0           = 1;
play.play_speed   = 2.0;

% --- pipeline ---
[sig_I, sig_Q] = fun.read_iq_float_interleaved(cfg.filename);

mag_am = fun.demod_am(sig_I, sig_Q);             %Calculate amplitude
mag_fm = fun.demod_fm_diffphase(sig_I, sig_Q);   %Calculate frequency

mag_am = mag_am(cfg.skip_samples:end);           %Crop
mag_fm = mag_fm(cfg.skip_samples:end);  

[sync_x, sync_y] = fun.find_sync_offsets_from_am( ...
    mag_am, cfg.width, cfg.height, cfg.manual_hsync, cfg.manual_vsync); %Synchronization

samples_per_frame = round(cfg.sampling_rate / cfg.frame_rate);

img_am = fun.reconstruct_frames_from_mag(mag_am, cfg.width, cfg.height, samples_per_frame, sync_x, sync_y);  %Reconstruction
img_fm = fun.reconstruct_frames_from_mag(mag_fm, cfg.width, cfg.height, samples_per_frame, sync_x, sync_y);

num_frames = min(size(img_am,3), size(img_fm,3));
img_am = img_am(:,:,1:num_frames);
img_fm = img_fm(:,:,1:num_frames);

img_fuse2 = fun.fuse_am_fm_gated_detail(img_am, img_fm, fuse);

fun.show_three_synced(img_am, img_fm, img_fuse2, cfg.frame_rate, play.k0, play.play_speed);

end
