import cv2
import numpy as np

img = cv2.imread("images/Podcast10_Picture_500x500.jpg")
print(img.shape)
imgResize = cv2.resize(img, (300, 200))
print(imgResize.shape)
#imgResize2 = cv2.resize(img, (3000, 2000))
#print(imgResize2.shape)
imgCropped = img[0:200,200:500]    #special. Height first, then width
cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
#cv2.imshow("Image Resize2", imgResize2)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)
