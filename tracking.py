#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import os

import lib

# option dictionary
opts = dict()

# video options
opts["video_file"] = "circular_scan_unprocessed.mp4"  # video file path
opts["ref_time"] = 0.043  # timestamp for reference frame in seconds

# preprocessing
opts["res"]                 = 500           # image output resolution (width)
opts["contrast_quant"]      = 0.95          # automatic contrast quantile
opts["mask_r"]              = 0.6           # circular mask radius, relative units
opts["mask_pos"]            = (0.5, 0.5)    # circular mask radius center position, relative units
opts["reflect_th"]          = 245           # reflection threshold for 8 bit (=maximum of 255)
opts["reflect_dilate"]      = 0.05          # dilation kernel size for reflection removal, relative units
opts["kernel_size"]         = 20 / 600      # high pass filter kernel size

# feature detection options
opts["fdet_args"]           = (500, )   # feature detection, parameters, first is max. number of keypoints
opts["min_matches"]         = 15        # minimum number of matches required 
opts["match_std_th"]        = 3         # maximum standard deviation in pixels for the matches
opts["max_matches_shown"]   = 30        # maximum number of displayed match connections


# open the video
video = lib.VideoStream(opts["video_file"], hsize=opts["res"])

# init preprocessor
proc_fdet = lib.Preprocessor(video.size, opts["kernel_size"], opts["contrast_quant"], 
                             opts["reflect_th"], opts["reflect_dilate"], opts["mask_r"], opts["mask_pos"])

# init feature matcher
fdet = lib.FeatureDetector(*opts["fdet_args"])

# get ref image for feature_detector
video.set_position(opts["ref_time"])
img_ref_fdet = proc_fdet.process(video.get_frame())
fdet.init_reference(img_ref_fdet)
video.set_position(0)

# init history image
img_hist = cv.cvtColor(img_ref_fdet, cv.COLOR_GRAY2BGR)

# open output window
lib.new_window("Tracking", [1200, 650])

# breaks when no video / video finished
while True:

    # get frame
    if (frame := video.get_frame()) is None:
        break
  
    # preprocessing and feature matching
    img_proc_fdet = proc_fdet.process(frame)
    fdet_success, shift, mask, good, kp1, kp2, matches_shown\
            = fdet.compute(img_proc_fdet, opts["min_matches"], 
                           opts["match_std_th"], opts["max_matches_shown"])

    # draw detected position in reference frame
    if fdet_success:
        # slight coloring depending on matching quality,
        # can be parametrized to show hugely different colors
        weight = min(1, 0.5 + 0.5*(np.count_nonzero(mask) - opts["min_matches"])\
                        /5/opts["min_matches"])
        color = (0, 80*weight, 255)
       
        lib.draw_found_position(img_hist, shift, color)
    
    # plot keypoints and matches
    img_curr_bgr = cv.cvtColor(img_proc_fdet, cv.COLOR_GRAY2BGR)
    img_comb = cv.drawMatches(img_hist, kp1, img_curr_bgr, kp2, good, None, matchColor=(0, 180, 250),
                              singlePointColor=(0, 160, 200), matchesMask=matches_shown, flags=0)
    
    # highlight OCT beam in current image      
    lib.draw_current_oct_beam(img_comb)

    # highlight OCT beam in reference image
    if fdet_success:
        lib.draw_oct_beam_reference(img_comb, img_ref_fdet, shift)

    # combine results with raw frame
    img_comb = np.hstack((img_comb, cv.cvtColor(frame.astype(np.uint8), cv.COLOR_GRAY2BGR)))
        
    # display everything
    cv.imshow("Tracking", img_comb)

    # Press Q on keyboard to exit, space to pause
    if lib.keyboard_interaction():
        break

# release the video capture
video.close()

# show all detected positions
cv.imshow('Tracking', img_hist)
lib.wait_for_exit('Tracking')

# close all windows
cv.destroyAllWindows()
