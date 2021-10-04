import cv2

"""
object_segmentation_webcam.py

This program captures video from webcam and performs object segmentation.
Object segmentation is the process of detecting and indentifying objects
in an image.
"""

thres = 0.45

#webcam capture settings
cap = cv2.VideoCapture(0) #which webcam
cap.set(3, 640)           #width
cap.set(4, 480)           #height
cap.set(10,70)            #brightness

className=[]              #all the classes (i.e. object types) that we detect
classFile = 'coco.names'
with open(classFile,'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" 
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath) #params are model, config
net.setInputSize(320,320)                 #set input size for frame 
net.setInputScale(1.0/127.5)              #set scalefactor value for frame
net.setInputMean((127.5, 127.5, 127.5))   #set mean value for frame
net.setInputSwapRB(True)                  #swap red and blue

while True:
  success, img = cap.read() 
  classIds, confs, bbox = net.detect(img, confThreshold=thres)

  if len(classIds) != 0:    #so that when the webcam captures nothing, we don't
                            # error out.
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
      cv2.putText(img, classNames[classId-1], (box[0]+10,box[1]+30), 
        cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,0), thickness=2)
      cv2.putText(img, str(round(confidence*100,2)), (box[0]+150,box[1]+30), 
        cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,0), thickness=2)

  cv2.imshow("Output", img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break;

#EOF
