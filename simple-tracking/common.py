import numpy as np
import cv2
import os

def show(image, delay=10, window='image'):
    '''Scales an image to a nice size and display it'''
    big = max(image.shape)
    scale = 800.0 / big
    small = cv2.resize(image, (0,0), fx = scale, fy = scale)
    print(small.shape)
    cv2.imshow(window, small)
    cv2.waitKey(delay)

def read_history(frames, frame_dir):
    '''Reads a number of frames into a history list and computes a baseline as the median image'''
    print(frame_dir)
    history = [read_and_blur(i, frame_dir)[0] for i in frames]
    baseline = np.median(history, 0)
    return(history, baseline)

def read_and_blur(frame_number, frame_dir):
    '''Read an image and get a blurred intensity frame from it'''
    image_name = '%s/f%05d.jpg' % (frame_dir, frame_number)
    img = cv2.imread(image_name)
    h = cv2.cvtColor(cv2.GaussianBlur(img, (51, 51), 0), cv2.COLOR_BGR2HSV).astype(np.float32)[:,:,2]
    return(h,img,image_name)

def versus_baseline(h, baseline):
    diff = h - baseline
    shapes = (diff*(h>(baseline+5)))
    scale = 250 / np.max(shapes)
    shapes = (scale * shapes).astype(np.uint8)
    return(shapes)

def setup_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 50
    params.maxThreshold = 200
    params.thresholdStep = 5
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
                      
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
                      
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    return(cv2.SimpleBlobDetector_create(params))
