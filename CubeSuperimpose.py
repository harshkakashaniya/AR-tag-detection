'''
/************************************************************************
 MIT License

 Copyright (c) 2018 Harsh Kakashaniya,Koyal Bhartia

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *************************************************************************/

/**
 *  @file    final.py
 *  @author  Harsh Kakashaniya and Koyal Bhartia
 *  @date    27/2/2019
 *  @version 1.0
 *
 *  @brief Project 1,AR tag detection
 *
 *  @section DESCRIPTION
 *
 *  This is code has 3 parts,
 1. tag id Tag_id_detection
 2. Superimposing image on AR tag with orientation
 3. Superimposing of cube on AR tag with orientation
 *
 */
'''
# Libraries
import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
from numpy import linalg as la
import math
from PIL import Image
import random


try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
#-------------------------------------------------------------------------------

# @brief Function for converting gray scale to binary
#
#  @param Matrix
#
#  @return Matrix
#
def binary(mat):
    for row in range(0,len(mat)):
        for col in range(0,len(mat[0])):
            if (mat[row,col]>150):
                mat[row,col]=1
            else:
                mat[row,col]=0
    return mat

# @brief To give 4 points in Edgedetection
#
#  @param Image and old_ctr points
#
#  @return corners
#
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
        #print(np.shape(ctr))
        old_ctr=ctr
    return ctr

# @brief Calculate homography
#
#  @param source and destination
#
#  @return homography
#
def homography_calc(src,des):
    #print(des)
    #print(np.shape(des))
    c1 = des[0,0]
    c2 = des[0,1]
    c3 = des[0,2]
    c4 = des[0,3]

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
# @brief To fix image on top of AR tag
#
#  @param image , corners , source
#
#  @return superimposed image
#

def Superimposing(ctr,image,src):
    pts_dst = np.array(ctr,dtype=float)
    pts_src = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float)
    h = homography_calc(pts_src, pts_dst)


    temp = cv2.warpPerspective(src, h,(image.shape[1],image.shape[0]));
    cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    image = image + temp;

    return image,h
# @brief To find get grid of tag image
#
#  @param image , corners
#
#  @return tag image and gray scale image
#
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
    return tag_image,warp2
# @brief To find tag ID and tag image
#
#  @param image , corners
#
#  @return tag and tag_id
#
def Tag_id_detection(ctr,tag_image):
    gray = cv2.cvtColor(tag_image,cv2.COLOR_BGR2GRAY)
    pixel_value=binary(gray)
    print(pixel_value,'Tag Value')
    status=0
    A_ctr=ctr[0][0]
    #print(A_ctr,'ctr A')
    B_ctr=ctr[0][1]
    #print(B_ctr,'ctr B')
    C_ctr=ctr[0][2]
    #print(C_ctr,'ctr B')
    D_ctr=ctr[0][3]
    #print(D_ctr,'ctr C')
    if (pixel_value[2,2] == 1):
        L1=A_ctr
        L2=B_ctr
        L3=C_ctr
        L4=D_ctr
        status=0
        one = pixel_value[4,4]
        two = pixel_value[4,3]
        three = pixel_value[3,3]
        four = pixel_value[3,4]

    elif pixel_value[5,2]==1:
        L1=D_ctr
        L2=A_ctr
        L3=B_ctr
        L4=C_ctr
        status=1
        one = pixel_value[3,4]
        two = pixel_value[4,4]
        three = pixel_value[4,3]
        four = pixel_value[3,3]

    elif pixel_value[5,5] == 1:
        L1=C_ctr
        L2=D_ctr
        L3=A_ctr
        L4=B_ctr
        status=2
        one = pixel_value[3,3]
        two = pixel_value[3,4]
        three = pixel_value[4,4]
        four = pixel_value[4,3]

    elif pixel_value[2,5] == 1:
        L1=B_ctr
        L2=C_ctr
        L3=D_ctr
        L4=A_ctr
        status=3
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

    #print(new_ctr,'new_ctr')

    tag_id = four*8 + three*4 + two*2 + one*1
    print('Tag id value will be',tag_id)
    return new_ctr,tag_id
# @brief To draw 3d cube
#
#  @param image , perspective points in camera frame
#
#  @return image of cube
#
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,0,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,150,150),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# @brief Calculates projection matrix
#
#  @param Homography
#
#  @return Projection matrix for 3D transformation
#
def Projection_mat(homography):
   # homography = homography*(-1)
   # Calling the projective matrix function
   K =np.array([[1406.08415449821,0,0],
       [ 2.20679787308599, 1417.99930662800,0],
       [ 1014.13643417416, 566.347754321696,1]])

   K=K.T
   rot_trans = np.dot(la.inv(K), homography)
   col_1 = rot_trans[:, 0]
   col_2 = rot_trans[:, 1]
   col_3 = rot_trans[:, 2]
   l = math.sqrt(la.norm(col_1, 2) * la.norm(col_2, 2))
   rot_1 = col_1 / l
   rot_2 = col_2 / l
   translation = col_3 / l
   c = rot_1 + rot_2
   p = np.cross(rot_1, rot_2)
   d = np.cross(c, p)
   rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   rot_3 = np.cross(rot_1, rot_2)

   projection = np.stack((rot_1, rot_2, rot_3, translation)).T
   return np.dot(K, projection)

    #def Three_d_cube(K, homography):
    #    return 0
# @brief it draws cube on image
#
#  @param Projection matrix and iamge
#
#  @return None
#
def Cube3D(proj_mat,image):
    axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    Proj= np.matmul(axis,proj_mat.T)
    # Normalize the matrix
    Norm1 = np.divide(Proj[0],Proj[0][2])
    Norm2 = np.divide(Proj[1],Proj[1][2])
    Norm3 = np.divide(Proj[2],Proj[2][2])
    Norm4 = np.divide(Proj[3],Proj[3][2])
    Norm5 = np.divide(Proj[4],Proj[4][2])
    Norm6 = np.divide(Proj[5],Proj[5][2])
    Norm7 = np.divide(Proj[6],Proj[6][2])
    Norm8 = np.divide(Proj[7],Proj[7][2])

    points = np.vstack((Norm1,Norm2,Norm3,Norm4,Norm5,Norm6,Norm7,Norm8))
    final_2d=np.delete(points,2, axis=1)
    draw(image,final_2d)
    return image
# @brief Main loop to run the code and video breaking
#
#  @param path of video, source of lena
#
#  @return image_array and size of image
#
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
        if (count==0):
            old_corners=0
        corners=Edgedetection(image,old_corners)
        #img = cv2.drawContours(image, corners,0,(0,255,0),1)
        if(len(corners)==0):
            corners=old_corners

        tag_image,Tag=perspective_for_tag(corners,image)
        new_corners,tag_id=Tag_id_detection(corners,tag_image)


        image,h=Superimposing(new_corners,image,src)
        proj_mat=Projection_mat(h)
        image=Cube3D(proj_mat,image)
        old_corners=corners
        count += 1
        print('Number of frames is',count)
        cv2.imwrite('%d.jpg' %count,image)
        img_array.append(image)
        success, image = vidObj.read()

    return img_array,size
#--------------------------------------------------------------
# @brief Loop to run video from image array
#
#  @param final image array and size
#
#  @return None
#
def video(img_array,size):
    video=cv2.VideoWriter('video_cube.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    #print(np.shape(img_array))
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
#---------------------------------------------------------------
# main
if __name__ == '__main__':

    # Calling the function
    src=cv2.imread('lena.jpg')
    #print(np.size(src))
    Image,size=Imageprocessor('Tag0.mp4',src)
    video(Image,size)
