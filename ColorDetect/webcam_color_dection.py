"""
Color Detection Project.

Uses webcam and detects certain colors (in this case, orange, pink and green)
and follows the path of the object (that has the color) and paints dots on the
canvas.
"""

import cv2
import numpy as np

#Set the dimensions of the canvas
frameWidth = 640
frameHeight = 480

#We will be using webcam video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

myColors = [[5, 107, 0, 19, 255, 255],      #detect orange    
            [133, 56, 0, 159, 156, 255],    #detect pink
            [57, 76, 0, 100, 255, 255]]     #detect green

#KK: Each item in myColors represents an array, which in turn
# represents the values hue_min, hue_max, sat_min, sat_max, val_min
# and val_max. We go these numbers by creating another script with
# trackbars and adjusting the trackbars until the only thing visible
# in an HSV image is the color/item we want to detect.

#Below in BGR format
myColorValues= [ [51,153,255],  #orange
                 [255,0,255],   #pink
                 [0,255,0] ]    #green

myPoints = [] #[x, y, colorId]

def findColor(img, myColors):
  """
  findColor(img, myColors)

  returns the coordinates and colorId for the detection in question  
  """
  imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  count = 0
  newPoints=[]
  #lower = np.array([h_min, s_min, v_min])
  #upper = np.array([h_max, s_max, v_max])
  for color in myColors:
    lower = np.array(color[0:3])
    upper = np.array(color[3:6])
    mask = cv2.inRange(imgHSV, lower, upper)  #the mask is an verion of the 
                                              # original image where everything
                                              # is blackend except for the 
                                              # color we wish to detect.
    x,y = getContours(mask)
    #place a color on canvas near the color detection points
    cv2.circle(imgResult,(x,y), 10, myColorValues[count], cv2.FILLED)
    if x !=0 and y != 0:
      newPoints.append([x, y, count])
    count += 1
    #cv2.imshow(str(color[0]), mask)
  return newPoints

def getContours(img):
  contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  x,y,w,h = 0,0,0,0
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
      cv2.drawContours(imgResult, cnt, -1, (255,0,0), 3)
      peri = cv2.arcLength(cnt,True)
      approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
      x, y, w, h = cv2.boundingRect(approx)
  return x+w//2, y

def drawOnCanvas(myPoints, myColorValues):
  """
  drawOnCanvas(myPoints, myColorValues)

    draws a circle following the movement path of item with the color 
    in question
  """
  for point in myPoints:
    cv2.circle(imgResult, (point[0],point[1]), 10, myColorValues[point[2]], cv2.FILLED)

while True:
  success, img = cap.read()
  imgResult = img.copy()
  newPoints = findColor(img, myColors)
  if len(newPoints) != 0:
    for newP in newPoints:
      myPoints.append(newP)
  if len(myPoints) != 0:
    drawOnCanvas(myPoints, myColorValues)
  cv2.imshow("Result", imgResult)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

#EOF
