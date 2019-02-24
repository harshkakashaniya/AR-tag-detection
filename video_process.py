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
def FrameCapture(path,template):

    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
    success = 1
    x_offset=y_offset=30
    img_array=[]
    lower = np.array([10, 20, 10], dtype = "uint8")
    upper = np.array([200, 200, 200], dtype = "uint8")
    sizeo=(1080,1920)
    filter=np.array(np.ones(sizeo), dtype = "uint8")

    b=0
    c=0
    d=0


    while (success and count<20):

        success, image = vidObj.read()

        height,width,layers=image.shape
        #print(height,width)
        size = (width,height)

        for i in range(height):
            for j in range(width):
                if( i<70 or i>height-70 or j<70 or j>width-70 ):
                    filter[i,j]=0


        #filtered = cv2.bitwise_and(image, image, mask = filter)
        mask = cv2.inRange(image, lower, upper)
        #print(np.shape(mask))
        corners = cv2.goodFeaturesToTrack(mask,50,0.01,1)
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

        corner_x_min_x=10000
        corner_x_min_y=10000
        corner_x_max_x=0
        corner_x_max_y=0
        corner_y_min_x=10000
        corner_y_min_y=10000
        corner_y_max_x=0
        corner_y_max_y=0

        for i in range (1,len(corners)):
            corner_x=corners[i,0,0]
            corner_y=corners[i,0,1]

            if(corner_x_min_x>corner_x and corner_x!=0):
                corner_x_min_x=corners[i,0,0]
                corner_x_min_y=corners[i,0,1]

            if(corner_x_max_x<corner_x and corner_x!=0):
                corner_x_max_x=corners[i,0,0]
                corner_x_max_y=corners[i,0,1]

            if(corner_y_min_y>corner_y and corner_y!=0):
                corner_y_min_x=corners[i,0,0]
                corner_y_min_y=corners[i,0,1]

            if(corner_y_max_y<corner_y and corner_y!=0):
                corner_y_max_x=corners[i,0,0]
                corner_y_max_y=corners[i,0,1]

        print(corner_x_min_x)
        print(corner_x_min_y)
        print(corner_x_max_x)
        print(corner_x_max_y)
        print(corner_y_min_x)
        print(corner_y_min_y)
        print(corner_y_max_x)
        print(corner_y_max_y)
        cv2.circle(image,(corner_x_min_x,corner_x_min_y),1,(0,255,0),5)
        cv2.circle(image,(corner_y_min_x,corner_y_min_y),1,(0,255,0),5)
        cv2.circle(image,(corner_x_max_x,corner_x_max_y),1,(255,0,0),5)
        cv2.circle(image,(corner_y_max_x,corner_y_max_y),1,(255,0,0),5)

        slope_max=math.atan(abs(corner_x_max_y-corner_y_max_y)/abs(corner_x_max_x-corner_y_max_x))*(7*180)/22
        slope_min=math.atan(abs(corner_x_min_y-corner_y_min_y)/abs(corner_x_min_x-corner_y_min_x))*(7*180)/22
        slope_avg=(slope_max+slope_min)/2

        aaa_max=str(round(slope_max,2))+'max_slope'
        aaa_min=str(round(slope_min,2))+'min_slope'
        aaa_avg=str(round(slope_avg,2))+'avg_slope'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,aaa_max,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,aaa_min,(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,aaa_avg,(10,300), font, 1,(0,0,0),2,cv2.LINE_AA)



        for i in corners:
            x,y = i.ravel()
            #cv2.circle(image,(x,y),3,(0,255,0),-1)
        count += 1

        print(count)
        image[corner_x_min_y:corner_x_min_y+template.shape[0], corner_x_min_x:corner_x_min_x+template.shape[0]] = template
        x_offset+=20

        img_array.append(image)

    return img_array,size

#--------------------------------------------------------------
'''
#superimpose
def Superimpose(img_array,template):
    for i in range(0,len(img_array)):
        #strr=str(i).join(['frame', '.jpg'])
        #print(strr)
        l = cv2.imread(img_array[i])
        #rotated = imutils.rotate(s, angle)


        x_offset+=20
        l[y_offset:y_offset+template.shape[0], x_offset:x_offset+template.shape[1]] = template
        #new=cv2.imwrite("newframe%d.jpg" % count, l)
        Super_Image.append(l)
        count+=1
        angle+=30
    return Super_Image
'''
#video file
def video(img_array,size):
    video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5.0,size)
    print(np.shape(img_array))
    for i in range(0,len(img_array)):
        video.write(img_array[i])
    video.release()
#---------------------------------------------------------------
# Driver Code
if __name__ == '__main__':

    # Calling the function
    template = cv2.imread('Lena.png',-1)
    resized=cv2.resize(template,dsize=None,fx=0.39,fy=0.39)
    Image,size=FrameCapture('Tag0.mp4',resized)
    #Super_Image=Superimpose(Image,template)
    video(Image,size)
