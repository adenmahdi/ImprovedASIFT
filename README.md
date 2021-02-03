# Improved ASIFT
By, Silver Lewis #101013172 & Mahdi Aden #100996245

### Summary: 

We are improving the ASIFT algorithm in order to both improve running time of the algorithm while ensuring the accuracy of and to better deal with panorama distortions. This algorithm will then be implemented in the use of automatic mosaic construction, specifically in the creation of indoor panoramas and a confirmation that the accuracy of the program is confirmed.
Background:

Affine Scale Invariant Feature Transformation, otherwise known as ASIFT, is an algorithm based on the SIFT algorithm used for image feature detection. ASIFT specifically seeks to assess whether two images include the same objects by comparing the feature vectors by first simulating possible camera axis orientation shifts and applying the SIFT algorithm to each possible shift. Compared to the original SIFT algorithm, proposed by Brian Lowe in 1999, which was only invariant with regards to zoom, translation and rotation, ASIFT is fully invariant with regards to any transformation.
  
However, given the increased computational complexity, the ASIFT algorithm is much slower than the SIFT algorithm and requires decent lighting conditions or rich textures to be effective. For indoor panoramas especially, where the camera’s roll is minimal, the algorithm is inefficient compared to the simpler counterpart. These problems can be solved through simplifications of the algorithms. These improvements we will build will be based on the article “An Improved ASIFT Algorithm for Indoor Panorama Image Matching”. This article details the steps required to improve ASIFT in regards to the shortcomings mentioned earlier.  No online implementation of the improvements are available and the algorithm comparisons done with their algorithm is minimal (Consisting of four garage panorama images).


### Challenge:

Although some of the steps provided in the “An Improved ASIFT Algorithm for Indoor Panorama Image Matching” can be done with OpenCV methods, the majority of the steps require direct manipulation of the ASIFT algorithm. Specifically the optimization of affine transformation comparison and certain steps of the new process for feature detection. ASIFT can still have some problems in its feature detection, further optimization not detailed in the paper is still potentially possible with panoramas. 

### Goals and Deliverables:

This project aims to recreate the algorithm described in “An Improved ASIFT Algorithm for Indoor Panorama Image Matching” By lowering our sample case to only panorama’s, we can optimize the ASIFT algorithm to increase its computational speed from quadratic to linear time. Improvements can also be made to the recognition software. and mimic its improvements to execution speed and feature detection.  

This project will then hope to compare  additional panorama scenes using the improved ASIFT algorithm. All the test images used in the paper involved garages. While this makes a compelling case, this project will hope to compare the algorithm with more varied environments than the research paper used. This will allow more exact results to be observed when compared to ASIFT outputs. This project will also attempt to aim at additional optimizations for panorama ASIFT feature detection. 

Success for this project will result in images returning similar features detected described in the article while maintaining an improved computational time. This initially will be done on the images provided in the article and then on additional panorama images.



### Schedule:

| Week: | Mahdi’s Tasks  | Silver’s Task  |
|---|---|---|
| Feb 6th  | Submit Proposal  | Submit Proposal  | 
| Feb 13th  | Read the SIFT and ASIFT algorithm/paper or  Find and read a  ASIFT implementation  | Read the SIFT and ASIFT algorithm/paper or Find and read a  ASIFT implementation  | 
| Feb 20th  | Find and collect some sample panoramas  | Find and collect some sample panoramas  | 
|  Feb 27th | Start on simplifying simulated transformations  | Let the panoramic image be projected into 7 normal perspective images | 
|  March 6th | End on simplifying simulated transformations. Start on dividing panoramic images | Gaussian filtering of the image after size adjustment to make the image fuzzy, in order to facilitate the following extraction of feature points simulation of the image in the X direction of the tilt transformation | 
|  March 13th | End on dividing panoramic images  | use SIFT algorithm for feature points recognition and matching   | 
|  March 20th | Find possible improvements for the algorithm | remove repeated and wrong matching pairs | 
|  March 27th | Implement possible improvements for the algorithm  | restore the image size and output matching results. Then re-project the normal perspective images return to the panoramic images | 
|  April 3rd | Final Report Recorded | Final Report Recorded | 
|  April 10th | Time is left here as buffer | Time is left here as buffer | 
|  April 14th | Time is left here as buffer | Time is left here as buffer | 
