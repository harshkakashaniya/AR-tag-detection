import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
import math
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

# Function to extract frames
def FrameCapture(path):

    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
    success = 1
    img_array=[]
    lower = np.array([10, 20, 10], dtype = "uint8")
    upper = np.array([200, 200, 200], dtype = "uint8")
    sizeo=(1080,1920)
    filter=np.array(np.ones(sizeo), dtype = "uint8")

    b=0
    c=0
    d=0


    while (success and count<450):

        success, image = vidObj.read()

        height,width,layers=image.shape
        #print(height,width)
        size = (width,height)
        '''
        for i in range(height):
            for j in range(width):
                if( i<70 or i>height-70 or j<70 or j>width-70 ):
                    filter[i,j]=0
        '''

        #filtered = cv2.bitwise_and(image, image, mask = filter)
        mask = cv2.inRange(image, lower, upper)
        #print(np.shape(mask))
        corners = cv2.goodFeaturesToTrack(mask,20,0.001,1)
        corners = np.int0(corners)

        j=0
        a=0
        while (j<len(mask[0])-10):
            for i in range(0,len(mask)-10):
                #print(i,j)
                if(mask[i,j]==0 and mask[i+10,j+10]==0 and a==0):
                    a=i
                    b=j
                    #print('got white point',a,b)
                if(mask[i,j]==0 and mask[i+10,j+10]==0):
                    c=i
                    d=j

            j=j+1

        cv2.rectangle(image,(b+100,a+20),(d-70,c-20),(0,255,0),1)
        for i in range (0,len(corners)):
            corner_x=corners[i,0,0]
            corner_y=corners[i,0,1]
            #print(corner_x,corner_y,'X and y dimension of corner.')
            if(corner_x<b+120 or corner_x>d-70 or corner_y<a+20 or corner_y>c):
                corners[i,0,0]=0
                #print(corner_x,corner_y, "got zero")
                corners[i,0,1]=0

        for i in corners:
            x,y = i.ravel()
            cv2.circle(image,(x,y),3,(0,255,0),-1)
        count += 1
        print(count)

        img_array.append(image)

    return img_array,size

#--------------------------------------------------------------
#video file
def video(img_array,size):
    video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25.0,size)
    print(np.shape(img_array))
    for i in range(0,len(img_array)):
        video.write(img_array[i])
    video.release()
#---------------------------------------------------------------
# Driver Code
if __name__ == '__main__':

    # Calling the function
    Image,size=FrameCapture('Tag0.mp4')
    video(Image,size)
