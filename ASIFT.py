#!/usr/bin/python3
import numpy as np
import cv2
from matplotlib import pyplot as plt
import Equirec2Perspec as E2P

def sift_alg(img1, img2, outImg, kp1, kp2):
	detector = cv2.SIFT_create()

	kp1, des1 = detector.detectAndCompute(img1, None)
	kp2, des2 = detector.detectAndCompute(img2, None)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	matches = bf.match(des1,des2)

	matches = sorted(matches, key = lambda x:x.distance)

	return matches

define ASIFT(img1, img2, kp1, kp2):
	num_matches=0
	for x in range(-5,5,0.1):
		z=x/(2*numpy.pi)
		for y in range(5,5,0.1):
			w=y/(2*numpy.pi)

			A=[[numpy.cos(z),-numpy.sin(z),0],[numpy.sin(z), numpy.cos(z),0],[0,0,1]]
			B=[[numpy.cos(w),-numpy.sin(w),0],[numpy.sin(w), numpy.cos(w),0],[0,0,1]]
			img1_warp = cv.warpPerspective(img1, A, (img1.shape[1], img1.shape[0]))
			img2_warp = cv.warpPerspective(img2, B, (img2.shape[1], img2.shape[0]))

			matches=sift_alg(img1_warp, img2_warp, outImg, kp1, kp2)
			if (len(matches)>num_matches):
				final=matches
				num_matches=len(matches)
	
	return matches
	
#created my own function to find points
def findPoints(image1, image2):
    gray1 = cv2.cvtColor(np.copy(image1), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(np.copy(image2), cv2.COLOR_BGR2GRAY)


    orb = cv2.ORB_create()#create orb to us in matcher
    keyPts1, desc1 = orb.detectAndCompute(gray1, None)#get key pts and descriptors 
    keyPts2, desc2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#initialize matcher
    matches = bf.match(desc1, desc2)#match descriptors
    sortedMatches = sorted(matches, key = lambda x:x.distance)#sort

    srcPts  = np.float32([keyPts1[m.queryIdx].pt for m in sortedMatches]).reshape(-1,1,2)
    dstPts  = np.float32([keyPts2[m.trainIdx].pt for m in sortedMatches]).reshape(-1,1,2)#return matched keypoints

    M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)#get homorapghy matrix
    h, w = image1.shape[:2]#hieght/width of img
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)#perspective transform
    
    h, w, x = image2.shape
    
    warpped = cv2.warpPerspective(image1, M, (w, h))#warp img to new perspective

    return warpped, dst

#def mergeImages():
#    leftRight, dst = findPoints(right,left)#gets matching points between right and left
#    cv2.imshow("Warped",cv2.cvtColor((leftRight),cv2.COLOR_BGR2GRAY))#Shows warped in greyscale
#    cv2.imshow("Merged",cv2.cvtColor((leftRight+left),cv2.COLOR_BGR2GRAY))#combine images in greyscale
#    cv2.waitKey()
#    cv2.destroyAllWindows()

def image_call(x,y, img1, img2, img1_1, img2_1):
	equ = E2P.Equirectangular(x)
	for(i in range(1,5)):
		img1_1= equ.GetPerspective(60,i*90, 0, img1.shape[1]/4, img1.shape[0])
	
	equ = E2P.Equirectangular(y)
	for(i in range(1,5)):
		img2_1= equ.GetPerspective(60,i*90, 0, img2.shape[1]/4, img2.shape[0])

print("Image 1: ")
x=input()
print("Image 2: ")
y=input()
img1 = cv2.imread(x,0)
img1_1=[]
img2_1=[]
image_call(x,y,img1, img2, img1_1,img2_1)
total_matches=[]

for (x in img1_1):
	num_matches=0
	for (y in img2_1):
		matches=ASIFT(x,y, kp1, kp2)
		if (len(matches)>num_matches):
			final=matches
			num_matches=len(matches)
