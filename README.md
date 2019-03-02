# Perception-project_1
The project focuses on detecting a custom AR Tag (a form of fiducial marker), that is used for
obtaining a point of reference in the real world, such as in augmented reality applications. The two aspects
to using an AR Tag: detection and tracking, has been implemented in this project. Following are the 2
stages:
* **Detection**: Detection involves finding the AR Tag from a given image sequence
* **Tracking**: Tracking involves keeping the tag in “view” throughout the sequence and performing image
processing operations based on the tag’s orientation and position (a.k.a. the pose).

Following is the reference AR Tag that has to be detected and tracked:

![](Images/ref_marker.png)

1 Detection of AR tag
**Encoding scheme**

AR Tags facilitate the appearance of virtual objects, games, and animations within the real world.  The
analysis of these tags can be done as followed.
* The tag has been decomposed into an 8*8 grid of squares, which includes a padding of 2 squares width
along the borders.  This allows easy detection of the tag when placed on white background.
* The inner 4*4 grid (i.e.  after removing the padding) has the orientation depicted by a white square in
the lower-right corner.  This represents the upright position of the tag.  This is different for each of the
tags provided in the different image sequences.
* Lastly, the inner-most 2*2 grid (i.e.  after removing the padding and the orientation grids) encodes the
binary representation of the tag’s ID, which is ordered in the clockwise direction from least significant
bit to most significant.  So, the top-left square is the least significant bit, and the bottom-left square is
the most significant bit.

Following are the AR Tags obtained on doing Perspective Transformation on image sequences.

![](Images/Tag1.jpg)
![](Images/Tag2.jpg)
![](Images/Tag3.jpg)

2 Process of Edge and Corner Detection
The process of the edge and corner detection has been implemented in the following way:
* The video stream is first converted into image frames.
* Detection is performed on each of the frames and then taking the fps as 25, the video is formed again.
* In the code, the function `Edgedetecion` has been scripted which takes in the image, the old corners that
can been computed and returns the new coordinates of the detected corners.  This is repeated for each
image frame.
* The Computer Vision methods made use of in the above implementation are:
  - cvtColor
  - medianBlur
  - threshold
  - findContours
  - arcLength
  - approxPolyDP
  - contourArea
  - isContourConvex
* Once the Corners are successfully detected,  the perspective transformation of the Tag is performed.
The function ”perspective for tag” has been scripted to get the transformed and resized tag image.
* After the above successful transformation the ID of the tag is obtained, the corners of the tag as well
as its ID with respect to its original orientation i.e compensated for any camera rotation is obatined.
* The above process has been written in Python using several functionalities of Computer Vision such
as Contours and Corner Detection algorithms.
* Thus this part of the project successfully identifies and detethe AR tag in any given image sequence.

Screenshot of Tag Identification and Edge Detection Output

![](Images/harsh.png)
![](Images/Tag_id.png)

3 Superimposing of image on AR tag

Homographic transformation is basically the transformation between two planes in projective space.
In augmented reality, it is the transformation that projects a square marker board from the world coordinate
system into a polygon in the image plane on the camera sensor (i.e., pixel coordinates).  In this project, once
the 4 corners of the tag is obtained with the Edge and corner detection done above, homography estimation
is performed.  This includes superimposing the image over the tag eg. the following image of lena.jpg

![](Images/lena.jpg)

The following steps has been successfully implemented in this part of the project:
* Homography between the corners of the template and the four corners of the tag has been computed
using the corners of the AR tag detected in the first part of the project.  The homography matrix has
been manually calculated in the code by finding the SVD of the world and camera coordinates system.
The SVD function of the Linear Algebra library from the Numpy library has been used to calculate
the same.
* Once the transformation of the template image has been sucessfully done above, tag has been done,
the tag is “replaced” with the transformed image.
* This also implies that the orientation of the transformed template image must match that of the tag at
any given frame, which has been taken care using efficient corner, contour and edge detection techniques
to get the correct orientation of the AR tag at any given time.

Output Images:

Following are the screenshots of the superimposed images of Lena.png on the AR Tag taken during
the  course  of  the  video.   The  images  show  that  the  superimposed  image  maintains  its  orientation,  thus
compensating for the rotation of the camera.

![](Images/Orientation1.jpg)
![](Images/Orientation2.jpg)
![](Images/Orientation3.jpg)
