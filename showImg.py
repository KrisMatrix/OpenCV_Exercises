import cv2
print("Package Imported:"+cv2.__version__)

img = cv2.imread("images/Podcast10_Picture_500x500.jpg")

cv2.imshow("Output", img)
cv2.waitKey(1000)
