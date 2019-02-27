import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
import math
from PIL import Image
import random
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
#-------------------------------------------------------------------------------

def Edgedetection(image,old_ctr):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray,3)
    (T, thresh) = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr=[]
    for j, cnt in zip(hierarchy[0], contours):
        cnt_len = cv2.arcLength(cnt,True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len,True)
        if cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and len(cnt) == 4  :
            cnt=cnt.reshape(-1,2)
            if j[0] == -1 and j[1] == -1 and j[3] != -1:
                ctr.append(cnt)
        print(np.shape(ctr))
        old_ctr=ctr
    return ctr

def Superimposing(ctr,image,src):
    print(ctr)
    pts_dst = np.array(ctr,dtype=float)
    pts_src = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float)

    h, status = cv2.findHomography(pts_src, pts_dst)

    print(image.shape[1],image.shape[0])

    temp = cv2.warpPerspective(src, h,(image.shape[1],image.shape[0]));
    cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    image = image + temp;

    return image

def perspectivetransform(ctr):
    dst1 = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M1 = cv2.getPerspectiveTransform(ctr, dst1)

    warp1 = cv2.warpPerspective(image.copy(), M1, (100,100))
    warp2 = warp1.copy()
    cv2.imshow("warp",warp1)
    cv2.imshow("corner detction",ar)

def Tag_id_detection(ctr):

    return 0

def Orientation(ctr):

    return 0

def Three_d_cube():

    return 0


#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
def Imageprocessor(path,src):

    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    img_array=[]

    while (success):
        if (count==0):
            success, image = vidObj.read()

        height,width,layers=image.shape
        size = (width,height)
        print(np.shape(image))
        if (count==0):
            old_corners=0
        if (count==0):
            old_corners=0
        corners=Edgedetection(image,old_corners)

        if(len(corners)==0):
            corners=old_corners
        image=Superimposing(corners,image,src)
        #perspective(corners)
        old_corners=corners
        count += 1
        print(count)
        #cv2.imwrite('%d.jpg' %count,edges)
        img_array.append(image)
        success, image = vidObj.read()

    return img_array,size
#--------------------------------------------------------------
#video file
def video(img_array,size):
    video=cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25.0,size)
    #print(np.shape(img_array))
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
#---------------------------------------------------------------
# main
if __name__ == '__main__':

    # Calling the function
    src=cv2.imread('lena.jpg')
    print(np.size(src))
    Image,size=Imageprocessor('Tag2.mp4',src)
    video(Image,size)
