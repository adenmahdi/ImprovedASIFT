#!/usr/bin/python3
import numpy
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import Equirec2Perspec as E2P
import math

pan1 = cv2.imread('C:\\Users\\matty\\Downloads\\ImprovedASIFT-main\\ImprovedASIFT-main\\IMG8362.jpg')
panumpyath1 = "C:\\Users\\matty\\Downloads\\ImprovedASIFT-main\\ImprovedASIFT-main\\IMG8362.jpg"

pan2 = cv2.imread("C:\\Users\\matty\\Downloads\\ImprovedASIFT-main\\ImprovedASIFT-main\\IMG8363.jpg")
panumpyath2 = "C:\\Users\\matty\\Downloads\\ImprovedASIFT-main\\ImprovedASIFT-main\\IMG8363.jpg"

def sift_alg(img1, img2):
	detector = cv2.SIFT_create()

	kp1, des1 = detector.detectAndCompute(img1, None)
	kp2, des2 = detector.detectAndCompute(img2, None)
	if (len(kp1)==0 or len(kp2)==0):
		print("No keypoints detected.")
		return [],[],[]
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	matches = bf.match(des1,des2)

	matches = sorted(matches, key = lambda x:x.distance)
	return matches, kp1, kp2

def ASIFT(img1, img2):
	num_matches=0
	for x in range(-1,1,1):
		z=x/(20*numpy.pi)
		finalMatches = []
		finalkp1 = []
		finalkp2 = []
		

		A=numpy.array([[numpy.cos(z),-numpy.sin(z),0],[numpy.sin(z), numpy.cos(z),0],[0,0,1]]).astype(numpy.float32)
		img1_warp = cv2.warpPerspective(img1, A, (img1.shape[1], img1.shape[0]))

		matches, kp1, kp2=sift_alg(img1_warp, img2)
		if (len(matches)>num_matches):
			finalMatches=matches
			finalkp1 = kp1
			finalkp2 = kp2
			num_matches=len(matches)
	return finalMatches, finalkp1, finalkp2
	
#roughly translates points found on the warped images back to the panoramas
def ImgPointsToPanoramaPoints(panaramaImg, imagePlace, kp):
	
	increment =720/2#images are 720 by 1080 but increments are 50% in order for neighbors to share objects
	zeroPlace = panaramaImg.shape[1]/2 - increment#first image center is panorama center
	imageStartPoint = imagePlace*increment*2+zeroPlace
	minYshift = (panaramaImg.shape[0]-1080)/2 +math.floor(panaramaImg.shape[0]/20)
	for i in range(len(kp)):
		kp[i].pt=((kp[i].pt[0]+imageStartPoint)%panaramaImg.shape[1], kp[i].pt[1]+minYshift)
	return kp


def SplitImage(img,path, min_size):
    equ = E2P.Equirectangular(path)
    imageWidth = 720
    amountOfImages = math.floor(min_size/imageWidth)
    splitImage= [None] * amountOfImages
    xMaps=[None] * amountOfImages
    yMaps=[None] * amountOfImages
    
    increment = 360/amountOfImages

    cv2.waitKey(0)
    
    for i in range(amountOfImages):
        splitImage[i], xMaps[i], yMaps[i]= equ.GetPerspective(60,i*increment, 0, 720, 1080)
        splitImage[i]= cv2.GaussianBlur(splitImage[i],(5,5),0)
    
    return equ, splitImage, xMaps, yMaps   		

min_size = min(pan1.shape[1], pan2.shape[1])

print("Image 1: ")
equ1, splitImage1, xMaps1, yMaps1 = SplitImage(pan1, panumpyath1, min_size)
equ2, splitImage2, xMaps2, yMaps2 = SplitImage(pan2, panumpyath2, min_size)
print("fin ")

#matches, kp1, kp2=ASIFT(splitImage1[0],splitImage2[0])

img3=255*numpy.ones(shape=[512, 512, 3], dtype=numpy.uint8)

#kp1=ImgPointsToPanoramaPoints(pan1, 0,kp1)
#kp2=ImgPointsToPanoramaPoints(pan2, 0,kp2)

#this will work as long as the panorama's are the same size and the imagePlace variable are the same as the kp
#img3=cv2.drawMatches(pan1,kp1,pan2,kp2,matches, img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite("C:\\Users\\matty\\Downloads\\ImprovedASIFT-main\\ImprovedASIFT-main\\Output.jpg", img3)

y=min(len(splitImage1), len(splitImage2))
total_matches_ASIFT=0

for x in range(y):
    matches, kp1, kp2=ASIFT(splitImage1[x],splitImage2[x])
    total_matches_ASIFT=len(matches)+total_matches_ASIFT
    if (len(matches)==0):
        continue
    else:
        kp1=ImgPointsToPanoramaPoints(pan1, x,kp1)
        kp2=ImgPointsToPanoramaPoints(pan2, x+1,kp2)
        
        img3=cv2.drawMatches(pan1,kp1,pan2,kp2,matches, img3)
        cv2.imwrite("C:\\Users\\matty\\Downloads\\ImprovedASIFT-main\\ImprovedASIFT-main\\Output"+str(x)+".jpg", img3)
        img3=255*numpy.ones(shape=[512, 512, 3], dtype=numpy.uint8)

print("Total ASIFT Matches: "+str(total_matches_ASIFT))