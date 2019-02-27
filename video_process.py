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

def Extractedges(image,lower,upper,dude_old):
    mask = cv2.inRange(image, lower, upper)

    #width, height, _ = image.shape
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(mask, 100, 200)
    contours, hierarchy=cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    i=0
    print(len(contours),'Initial')
    print(np.shape(contours[1]))
    while (i<len(contours)):
        h,w,t=np.shape(contours[i])
        print(h)
        if (h<100):
            print(i)
            del contours[i]
            i-=1
        i+=1

    print(len(contours),'Initial in second round')
    #print(np.shape(contours[1]))

    max=0
    i=0
    while (i<len(contours)):
        h,w,t=np.shape(contours[i])
        if (max<h):
            max=h
            print(max,'max print ho raha he')
        i+=1
    i=0
    threshold=max-150
    while (i<len(contours)):
        h,w,t=np.shape(contours[i])
        if (threshold<h):
            del contours[i]
            i-=1
            print('ud gaya sala')
        i+=1
    print(len(contours),'Initial in third round')


    print(len(contours),'Final')

    dude=[]
    AR_tag=0
    while(AR_tag!=1):
        min_top=10000
        for i in range(len(contours)):
            reshape=contours[i].reshape(len(contours[i]),2)
            x_cord=reshape[:,0]
            print(x_cord)
            top_point=int(np.min(x_cord))
            if(top_point<min_top):
                min_top=top_point

        for i in range(len(contours)):
            reshape=contours[i].reshape(len(contours[i]),2)
            x_cord=reshape[:,0]
            print(x_cord)
            top_point=int(np.min(x_cord))
            if(top_point==min_top):
                dude=contours[i]
                print('Found one')

        max_area=0
        for i in range(len(contours)):
            if(max_area<cv2.contourArea(contours[i])):
                max_area=cv2.contourArea(contours[i])
                print(cv2.contourArea(contours[i]),'Area %d' %i )

        for i in range(len(contours)):
            if(max_area-100<cv2.contourArea(dude)):
                dude=contours[i]
                AR_tag=1
            else:
                del contours[i]
                AR_tag=0

    '''
    if (len(dude)==0):
        dude_new=dude_old
    else :
        dude_new=dude.reshape(len(dude),2)

    for i in dude_new:
        x,y = i.ravel()
        cv2.circle(image,(x,y),5,(0,random.random()*255,0),-1)


        dude_old=dude_new


    return image,dude_new,dude_old

def cornerpoint(a_x_old,a_y_old,edge_matrix,threshold):
    difference_x=abs(a_x_old-edge_matrix[:,0])
    target=int(np.argmin(difference_x))
    x_deviation=abs(a_x_old-edge_matrix[int(np.argmin(difference_x)),0])
    y_deviation=abs(a_y_old-edge_matrix[int(np.argmin(difference_x)),1])


    if(x_deviation<threshold and y_deviation<threshold):
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
        corner_x=corners[i,0]
        corner_y=corners[i,1]

        if(corner_x_min_x>corner_x and corner_x!=0):
            corner_x_min_x=corners[i,0]
            corner_x_min_y=corners[i,1]

        if(corner_x_max_x<corner_x and corner_x!=0):
            corner_x_max_x=corners[i,0]
            corner_x_max_y=corners[i,1]

        if(corner_y_min_y>corner_y and corner_y!=0):
            corner_y_min_x=corners[i,0]
            corner_y_min_y=corners[i,1]

        if(corner_y_max_y<corner_y and corner_y!=0):
            corner_y_max_x=corners[i,0]
            corner_y_max_y=corners[i,1]

    edge_matrix=np.mat(([corner_x_min_x,corner_x_min_y],[corner_x_max_x,corner_x_max_y],
    [corner_y_min_x,corner_y_min_y],[corner_y_max_x,corner_y_max_y]))

    print(edge_matrix)

    if(count==0):
        a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old=initializecorners(edge_matrix)
    threshold=30
    #a_x,a_y=cornerpoint(a_x_old,a_y_old,edge_matrix,threshold)
    #b_x,b_y=cornerpoint(b_x_old,b_y_old,edge_matrix,threshold)
    #c_x,c_y=cornerpoint(c_x_old,c_y_old,edge_matrix,threshold)
    #d_x,d_y=cornerpoint(d_x_old,d_y_old,edge_matrix,threshold)
    a_x=edge_matrix[0,0]
    a_y=edge_matrix[0,1]
    b_x=edge_matrix[2,0]
    b_y=edge_matrix[2,1]
    c_x=edge_matrix[1,0]
    c_y=edge_matrix[1,1]
    d_x=edge_matrix[3,0]
    d_y=edge_matrix[3,1]
    a_x_old=a_x
    b_x_old=b_x
    c_x_old=c_x
    d_x_old=d_x

    cv2.circle(image,(a_x,a_y),1,(0,0,255),8)
    cv2.circle(image,(b_x,b_y),1,(0,0,255),8)
    cv2.circle(image,(c_x,c_y),1,(255,0,0),8)
    cv2.circle(image,(d_x,d_y),1,(255,0,0),8)
    cv2.circle(image,(edge_matrix[0,0],edge_matrix[0,1]),1,(0,0,0),3)
    cv2.circle(image,(edge_matrix[1,0],edge_matrix[1,1]),1,(0,0,0),3)
    cv2.circle(image,(edge_matrix[2,0],edge_matrix[2,1]),1,(0,0,0),3)
    cv2.circle(image,(edge_matrix[3,0],edge_matrix[3,1]),1,(0,0,0),3)


    pts_dst = np.array([[a_x, a_y], [b_x, b_y], [c_x, c_y],[d_x, d_y]])
    pts_src = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float)

    h, status = cv2.findHomography(pts_src, pts_dst)

    print(image.shape[1],image.shape[0])
    temp = cv2.warpPerspective(src, h, (image.shape[1],image.shape[0]))
    cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    image = image + temp;

    slope_max=math.atan(abs(b_y-a_y)/abs(b_x-a_x))*(7*180)/22
    slope_min=math.atan(abs(c_y-d_y)/abs(c_x-d_x))*(7*180)/22
    slope_avg=(slope_max+slope_min)/2

    print(slope_avg,'Slope')
    angle_max=str(round(slope_max,2))+'max_slope'
    angle_min=str(round(slope_min,2))+'min_slope'
    angle_avg=str(round(slope_avg,2))+'avg_slope'

    return image,angle_max,angle_min,angle_avg,edge_matrix,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old,h

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


def transform_mat(h,image,corner_x_min_x,corner_x_min_y,corner_y_min_x,corner_y_min_y,corner_y_max_x,corner_y_max_y):
    Kdash =[[1406.08415449821,0,0], [2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]
    K=np.transpose(Kdash)
    #pts_dst = np.array([[a_x, a_y], [b_x, b_y], [c_x, c_y],[d_x, d_y]])
    a=corner_x_min_x
    b=corner_x_min_y
    pts_src = np.array([[a+0,b+0],[a+50, b+0],[a+25,b+25],[a-25,b+25],[a+0,b-50],[a+50,b-50],[a+25,b-25],[a-25,b-25]],dtype=float)
    width=np.sqrt(math.pow(corner_x_min_x-corner_y_min_x,2)+math.pow(corner_x_min_y-corner_y_min_y,2))
    height=np.sqrt(math.pow(corner_x_min_x-corner_y_max_x,2)+math.pow(corner_x_min_y-corner_y_max_y,2))

    src=draw(image,pts_src)
    return src

def printslope(image,angle_max,angle_min,angle_avg,corners):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,angle_max,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,angle_min,(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,angle_avg,(10,300), font, 1,(0,0,0),2,cv2.LINE_AA)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(image,(x,y),1,(0,255,0),-1)

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
    #filter=np.array(np.ones(sizeo), dtype = "uint8")

    b=0
    c=0
    d=0

    while (success):
        if (count==0):
            success, image = vidObj.read()

        height,width,layers=image.shape
        size = (width,height)
        if (count==0):
            old_corners=0

        image,corners,old_corners=Extractedges(image,lower,upper,old_corners)

        if(count==0):
            a_x_old=0
            b_x_old=0
            c_x_old=0
            d_x_old=0
            a_y_old=0
            b_y_old=0
            c_y_old=0
            d_y_old=0


        image,angle_max,angle_min,angle_avg,edge_matrix,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old,h=edgesplotter(src,image,corners,count,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old,d_x_old,d_y_old)

        image=transform_mat(h,image,a_x_old,a_y_old,b_x_old,b_y_old,c_x_old,c_y_old)


        image=printslope(image,angle_max,angle_min,angle_avg,corners)

        count += 1
        print(count)
        #cv2.imwrite('%d.jpg' %count,image)
        img_array.append(image)
        success, image = vidObj.read()

    return img_array,size
#--------------------------------------------------------------
#video file
def video(img_array,size):
    video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25.0,size)
    #print(np.shape(img_array))
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
#---------------------------------------------------------------
# main
if __name__ == '__main__':

    # Calling the function
    src=cv2.imread('lena.jpg')
    #src=cv2.resize(src_raw, dsize=None, fx=0.25, fy=0.25)
    Image,size=Imageprocessor('Tag0.mp4',src)
    video(Image,size)
    '''
