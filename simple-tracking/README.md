# Old School Object Tracking

The idea here is that this project uses some simple ideas like blurring and background subtraction to find moving cars on the track. This could be used to build training data for a more advanced tracker using, possibly, convolutional neural networks.

Programs here include:

build-figures.py - Run this to build some figures that show off how the process works

track-1.py - Run this to process a bunch of images to get points of interest that may be moving cars. There are a fair number of false positives, but it is a fair bet that if exactly four objects are found, then they are all cars. Looking at several hundred sample frames, it appears that about 1/3 of all frames have exactly four found objects.

common.py - Common code for reading frames, blurring them, finding a baseline image and so on.

