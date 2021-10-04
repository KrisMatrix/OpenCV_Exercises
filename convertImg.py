import numpy as np
import cv2

img = cv2.imread("images/Podcast10_Picture_500x500.jpg")
kernel = np.ones((5,5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
imgCanny = cv2.Canny(img, 100, 100)
imgCanny2 = cv2.Canny(img, 150, 200)
imgDialation = cv2.dilate(imgCanny2, kernel, iterations=1) 
imgEroded = cv2.erode(imgDialation, kernel, iterations=1) 

cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blur Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Canny Image2",imgCanny2)
cv2.imshow("Dilation Image2",imgDialation)
cv2.imshow("Eroded Image2",imgEroded)
cv2.waitKey(0)
