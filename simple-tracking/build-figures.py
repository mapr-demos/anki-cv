import numpy as np
import cv2
import common

# Recreate the figures from the gdoc README
history, baseline = common.read_history(xrange(9898,9998,10), '/Users/tdunning/frames')
h, img, name = common.read_and_blur(9998, '/Users/tdunning/frames')
shapes = common.versus_baseline(h, baseline)
keys = common.setup_detector().detect(255-shapes)
xi = cv2.drawKeypoints(img, keys, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('im-1.jpg', img[500:1100,0:700,:])
cv2.imwrite('im-2.jpg', h[500:1100,0:700].astype(np.uint8))
cv2.imwrite('im-3.jpg', baseline[500:1100,0:700].astype(np.uint8))
cv2.imwrite('im-4.jpg', shapes[500:1100,0:700])
cv2.imwrite('im-5.jpg', xi[500:1100,0:700])
