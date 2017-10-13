import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Harris Corner Detector
def HCD(gray,height,width):

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    print "sobelx"
    print sobelx
    print "sobely"
    print sobely

    Ix2 = sobelx ** 2
    Iy2 = sobely ** 2
    Ixy = sobelx * sobely

    print "Ix2"
    print Ix2
    print "Iy2"
    print Iy2
    print "Ixy"
    print Ixy
    print "---"

    Ix2 = cv2.GaussianBlur(Ix2,(5,5),0)
    Iy2 = cv2.GaussianBlur(Iy2,(5,5),0)
    Ixy = cv2.GaussianBlur(Ixy,(5,5),0)
    print "Ix2"
    print Ix2
    print "Iy2"
    print Iy2
    print "Ixy"
    print Ixy
    
    print "---"
    print Ix2.shape
    print Iy2.shape
    print Ixy.shape

    height_i, width_i, depth_i = Ix2.shape

    for x in range(width_i):
        for y in range(height_i):
#print "x: %d --- y: %d" % (x,y)
            ix2 = Ix2[x][y]
            iy2 = Iy2[x][y]
            ixy = Ixy[x][y]
            H = np.array([[ix2, ixy], [ixy, iy2]])
    print H



filename = 'test2.jpg'
img = cv2.imread(filename)
print img
print img.shape
height, width, depth = img.shape
print height, width


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('dst',gray)

gray = np.float32(gray)
print "gray: %s" % len(gray)
print gray
HCD(gray, height, width)
#dst = cv2.cornerHarris(gray,2,3,0.04)
#print "dst"
#print dst

'''
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
'''

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
