import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
import math
from PIL import Image
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2



def Extractedges(image,lower,upper):
    mask = cv2.inRange(image, lower, upper)
    #print(np.shape(mask)
    '''
    for i in range(height):
        for j in range(width):
            if( i<70 or i>height-70 or j<70 or j>width-70 ):
                filter[i,j]=0
    filtered = cv2.bitwise_and(image, image, mask = filter)
    '''
    corners = cv2.goodFeaturesToTrack(mask,50,0.001,1)
    corners = np.int0(corners)

    #cv2.namedWindow("images_mask", cv2.WINDOW_NORMAL)
    #cv2.imshow("images_mask", mask)
    #cv2.waitKey()
    j=50
    a=0
    while (j<len(mask[0])-50):
        for i in range(50,len(mask)-50):
            #print(i,j)
            if(mask[i,j]==0 and mask[i+5,j+5]==0 and mask[i+10,j+10]==0 and a==0):
                a=i
                b=j
                #print('got white point',a,b)
            if(mask[i,j]==0 and mask[i+5,j+5]==0 and mask[i+10,j+10]==0):
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
    return image,corners

def cornerpoint(a_x_old,a_y_old,edge_matrix):
    difference_x=abs(a_x_old-edge_matrix[:,0])
    y_deviation=abs(a_y_old-edge_matrix[int(np.argmin(difference_x)),1])
    if(min(difference_x)<10 and y_deviation<10):
        a_x=edge_matrix[int(np.argmin(difference_x)),0]
        a_y=edge_matrix[int(np.argmin(difference_x)),1]
    else:
        a_x=a_x_old
        a_y=a_y_old
    return a_x ,a_y

def initializecorners(edge_matrix):
    a_x_old=edge_matrix[0,0]
    a_y_old=edge_matrix[0,1]
    b_x_old=edge_matrix[2,0]
    b_y_old=edge_matrix[2,1]
    c_x_old=edge_matrix[1,0]
    c_y_old=edge_matrix[1,1]
    d_x_old=edge_matrix[3,0]
    d_y_old=edge_matrix[3,1]

    return a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old

def edgesplotter(src,image,corners,count,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old):
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

    edge_matrix=np.mat(([corner_x_min_x,corner_x_min_y],[corner_x_max_x,corner_x_max_y],
    [corner_y_min_x,corner_y_min_y],[corner_y_max_x,corner_y_max_y]))

    print(edge_matrix)

    if(count==0):
        a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old=initializecorners(edge_matrix)

    a_x,a_y=cornerpoint(a_x_old,a_y_old,edge_matrix)
    b_x,b_y=cornerpoint(b_x_old,b_y_old,edge_matrix)
    c_x,c_y=cornerpoint(c_x_old,c_y_old,edge_matrix)
    d_x,d_y=cornerpoint(d_x_old,d_y_old,edge_matrix)

    a_x_old=a_x
    b_x_old=b_x
    c_x_old=c_x
    d_x_old=d_x

    cv2.circle(image,(a_x,a_y),1,(0,255,0),5)
    cv2.circle(image,(b_x,b_y),1,(0,255,0),5)
    cv2.circle(image,(c_x,c_y),1,(255,0,0),5)
    cv2.circle(image,(d_x,d_y),1,(255,0,0),5)

    #pts_dst = np.array([[a_x, a_y], [b_x, b_y], [c_x, c_y],[d_x, d_y]])
    #pts_src = np.array([[0,0],[199, 0],[199, 199],[0,199]],dtype=float);

    #pts_dst = np.array([[200,200], [400,200], [400,400],[400,400]])

    #h, status = cv2.findHomography(pts_src, pts_dst)
    #print(h)
    #print(pts_dst)
    #print(image.shape[1],image.shape[0])
    #temp = cv2.warpPerspective(src, h, (image.shape[1],image.shape[0]))
    #cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    #image = image + temp;

    slope_max=math.atan(abs(b_y-a_y)/abs(b_x-a_x))*(7*180)/22
    slope_min=math.atan(abs(c_y-d_y)/abs(c_x-d_x))*(7*180)/22
    slope_avg=(slope_max+slope_min)/2

    print(slope_avg,'Slope')
    angle_max=str(round(slope_max,2))+'max_slope'
    angle_min=str(round(slope_min,2))+'min_slope'
    angle_avg=str(round(slope_avg,2))+'avg_slope'

    return image,angle_max,angle_min,angle_avg,edge_matrix,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old

def printslope(image,angle_max,angle_min,angle_avg,corners):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,angle_max,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,angle_min,(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,angle_avg,(10,300), font, 1,(0,0,0),2,cv2.LINE_AA)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(image,(x,y),3,(0,255,0),-1)
    return image

#-------------------------------------------------------------------------------
def Imageprocessor(path,src):

    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
    success = 1
    img_array=[]
    lower = np.array([10,10, 10], dtype = "uint8")
    upper = np.array([200, 200, 200], dtype = "uint8")
    sizeo=(1080,1920)
    filter=np.array(np.ones(sizeo), dtype = "uint8")

    b=0
    c=0
    d=0

    while (success and count<500):

        success, image = vidObj.read()

        height,width,layers=image.shape
        size = (width,height)

        image,corners=Extractedges(image,lower,upper)

        if(count==0):
            a_x_old=0
            b_x_old=0
            c_x_old=0
            d_x_old=0
            a_y_old=0
            b_y_old=0
            c_y_old=0
            d_y_old=0


        image,angle_max,angle_min,angle_avg,edge_matrix,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old=edgesplotter(src,image,corners,count,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old)

        cv2.imwrite('%d.jpg' %count,image)

        image=printslope(image,angle_max,angle_min,angle_avg,corners)

        count += 1
        print(count)

        img_array.append(image)

    return img_array,size
#--------------------------------------------------------------
#video file
def video(img_array,size):
    video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5.0,size)
    print(np.shape(img_array))
    for i in range(0,len(img_array)):
        video.write(img_array[i])
    video.release()
#---------------------------------------------------------------
# main
if __name__ == '__main__':

    # Calling the function
    src=cv2.imread('candy.jpg')
    #src=cv2.resize(src_raw, dsize=None, fx=0.25, fy=0.25)
    Image,size=Imageprocessor('Tag0.mp4',src)
    video(Image,size)
