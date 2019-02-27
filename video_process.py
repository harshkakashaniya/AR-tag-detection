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

def binary(A):
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            if (A[i,j]>150):
                A[i,j]=1
            else:
                A[i,j]=0
    return A

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

def perspective_for_tag(ctr,image):
    dst1 = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]], dtype = "float32")

    M1,status = cv2.findHomography(ctr[0], dst1)
    warp1 = cv2.warpPerspective(image.copy(), M1, (100,100))
    warp2=cv2.medianBlur(warp1,3)
    #warp2= warp1-warp1_5

    tag_image=cv2.resize(warp2, dsize=None, fx=0.08, fy=0.08)

    return tag_image

def homography_calc(src,dest):
    c1 = tag_des[0]
    c2 = tag_des[1]
    c3 = tag_des[2]
    c4 = tag_des[3]

    w1 = src[0]
    w2 = src[1]
    w3 = src[2]
    w4 = src[3]

    A=np.array([[w1[0],w1[1],1,0,0,0,-c1[0]*w1[0],-c1[0]*w1[1],-c1[0]],
                [0,0,0,w1[0], w1[1],1,-c1[1]*w1[0],-c1[1]*w1[1],-c1[1]],
                [w2[0],w2[1],1,0,0,0,-c2[0]*w2[0],-c2[0]*w2[1],-c2[0]],
                [0,0,0,w2[0], w2[1],1,-c2[1]*w2[0],-c2[1]*w2[1],-c2[1]],
                [w3[0],w3[1],1,0,0,0,-c3[0]*w3[0],-c3[0]*w3[1],-c3[0]],
                [0,0,0,w3[0], w3[1],1,-c3[1]*w3[0],-c3[1]*w3[1],-c3[1]],
                [w4[0],w4[1],1,0,0,0,-c4[0]*w4[0],-c4[0]*w4[1],-c4[0]],
                [0,0,0,w4[0], w4[1],1,-c4[1]*w4[0],-c4[1]*w4[1],-c4[1]]])

    #Performing SVD
    u, s, vt = la.svd(A)

            # normalizing by last element of v
            #v =np.transpose(v_col)
    v = vt[8:,]/vt[8][8]

    req_v = np.reshape(v,(3,3))

    return req_v

def Tag_id_detection(ctr,tag_image):
    gray = cv2.cvtColor(tag_image,cv2.COLOR_BGR2GRAY)
    pixel_value=binary(gray)
    A_ctr=ctr[0][0]
    print(A_ctr,'ctr A')
    B_ctr=ctr[0][1]
    print(B_ctr,'ctr B')
    C_ctr=ctr[0][2]
    print(C_ctr,'ctr B')
    D_ctr=ctr[0][3]
    print(D_ctr,'ctr C')
    if (pixel_value[2,2] == 1):
        L1=A_ctr
        L2=B_ctr
        L3=C_ctr
        L4=D_ctr

        one = pixel_value[4,4]
        two = pixel_value[4,3]
        three = pixel_value[3,3]
        four = pixel_value[3,4]

    elif pixel_value[5,2]==1:
        L1=D_ctr
        L2=A_ctr
        L3=B_ctr
        L4=C_ctr
        one = pixel_value[3,4]
        two = pixel_value[4,4]
        three = pixel_value[4,3]
        four = pixel_value[3,3]

    elif pixel_value[5,5] == 1:
        L1=C_ctr
        L2=D_ctr
        L3=A_ctr
        L4=B_ctr
        one = pixel_value[3,3]
        two = pixel_value[3,4]
        three = pixel_value[4,4]
        four = pixel_value[4,3]

    elif pixel_value[2,5] == 1:
        L1=B_ctr
        L2=C_ctr
        L3=D_ctr
        L4=A_ctr
        one = pixel_value[4,3]
        two = pixel_value[3,3]
        three = pixel_value[3,4]
        four = pixel_value[4,4]

    else:
        L1=A_ctr
        L2=B_ctr
        L3=C_ctr
        L4=D_ctr

        one = pixel_value[4,4]
        two = pixel_value[4,3]
        three = pixel_value[3,3]
        four = pixel_value[3,4]


    new_ctr=np.array([[L1,L2,L3,L4]])

    print(new_ctr,'new_ctr')

    tag_id = four*8 + three*4 + two*2 + one*1

    return new_ctr,tag_id


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
        corners=Edgedetection(image,old_corners)
        if(len(corners)==0):
            corners=old_corners

        tag_image=perspective_for_tag(corners,image)
        new_corners,tag_id=Tag_id_detection(corners,tag_image)

        image=Superimposing(new_corners,image,src)
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
    Image,size=Imageprocessor('Tag0.mp4',src)
    video(Image,size)
