import numpy as np
import cv2
import os
import common

# this code reads images from a frame capture directory and prints a summary about
# the moving objects found. This summary can be used to drive a learning process
# to learn more robust models

frame_dir = '/Users/tdunning/frames'

# collect a historical record of multiple images to get a base image with
history, baseline = common.read_history(xrange(8900,9000,10), frame_dir)

# build us a blob detector
detector = common.setup_detector()

# now process real images
for i in xrange(9000,9999):
    h,img,image_name = common.read_and_blur(i, frame_dir)

    # every 10th frame goes into the history and updates the baseline
    if (i%10) == 0:
        history.append(h)
        if (len(history) > 10):
            history = history[len(history)-10:]
        baseline = np.median(history, 0)

    # compute the difference image, scale and convert back to nice form
    shapes = common.versus_baseline(h, baseline)

    # note that the blob detector much prefers dark on light blobs
    keys = detector.detect(255-shapes)

    # textual summary
    key_points = [(list(x.pt) + [x.size]) for x in keys]
    print("%s,%d,%s" % (os.path.basename(image_name), len(key_points), key_points))

    
