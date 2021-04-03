#!/usr/bin/python3
import numpy as np
import cv2
from matplotlib import pyplot as plt

def sift_alg(img1, img2, outImg):
	detector = cv2.SIFT_create()

	keypoints1, des1 = detector.detectAndCompute(img1, None)
	keypoints2, des2 = detector.detectAndCompute(img2, None)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	matches = bf.match(des1,des2)

	matches = sorted(matches, key = lambda x:x.distance)

	outImg=cv.drawMatches(img1, keypoints1, img2, keypoints2, matches)

	return matches

define ASIFT(img1, img2):
	num_matches=0
	for x in range(-5,5,0.1):
		z=x/(2*numpy.pi)
		for y in range(5,5,0.1):
			w=y/(2*numpy.pi)

			A=[[numpy.cos(z),-numpy.sin(z),0],[numpy.sin(z), numpy.cos(z),0],[0,0,1]]
			B=[[numpy.cos(w),-numpy.sin(w),0],[numpy.sin(w), numpy.cos(w),0],[0,0,1]]
			img1_warp = cv.warpPerspective(img1, A, (img1.shape[1], img1.shape[0]))
			img2_warp = cv.warpPerspective(img2, B, (img2.shape[1], img2.shape[0]))

			matches=sift_alg(img1_warp, img2_warp, outImg)
			if (len(matches)>num_matches):
				final=outImg
				num_matches=len(matches)

	cv.imwrite("Output.jpg", final)

print("Image 1: ")
x=input()
print("Image 2: ")
y=input()
img1 = cv2.imread(x,0)
dst = cv2.Mat();
// You can try more different parameters
rect = cv2.Rect(100, 100, 200, 200);
dst = src.roi(rect);
img1_1, img1_2, img1_3, img1_4 = cv2.split(img1)
img2 = cv2.imread(y,0)
img1_1, img1_2, img1_3, img1_4 = cv2.split(img1)
ASIFT(img1, img2)
