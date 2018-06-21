#!/usr/bin/env python 
import numpy as np
import cv2
import Obj_utilscv
import Obj_recog as orec
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 12

img = cv2.imread('aaaa.PNG')          # queryImage
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img2 = cv2.imread('box_in_scene.png',0) # trainImage
camera = cv2.VideoCapture(0)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()#800
surf = cv2.xfeatures2d.SURF_create(1100)
# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=15)#5
search_params = dict(checks=25)#50
# Auxiliary function to draw text with contrast in an image:
def draw_str(dst, x_y, s):
    x,y = x_y
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2, lineType=cv2.LINE_AA)
while(1):

        ret, frame = camera.read()
        frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        kp2, des2 = surf.detectAndCompute(frame,None)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        #matchesMask = [[0, 0] for i in range(len(matches))]
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.60*n.distance: # 0.7, 0.75
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img2.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        # Get descriptor dimension of each feature:
        if des2 is not None:
            if len(des2) > 0:
                dim = len(des2[0])
            else:
                dim = -1
        draw_str(frame, (20, 20),"Method {0}, {1} features found, desc. dim. = {2} ".format('SURF', len(kp2), dim))
        draw_str(frame, (50, 50), "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color matchColor = (0,255,0)
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers matchesMask = matchesMask
                           flags = 2)

        imgout = cv2.drawMatches(img2,kp1,frame,kp2,good,None,**draw_params) #img3 = cv2.drawMatches(img2,kp1,frame,kp2,good,None,**draw_params)
        cv2.imshow('view', imgout)

        if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        #  plt.imshow(img3, 'gray'),plt.show()
