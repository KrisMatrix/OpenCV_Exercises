import cv2
import mediapipe as mp
import time

class poseDetector():
  """
  class poseDetector()

  A class to set up pose detection.
  """
  def __init__(self, mode=False, 
                #upBody=False,  #seems like this feature is no longer available
                                # in mediapipe mpPose.Pose.
                smooth=True,
                detectionCon=0.5, 
                trackCon=0.5):
    self.mode = mode
    #self.upBody = upBody
    self.smooth = smooth
    self.detectionCon = detectionCon
    self.trackCon = trackCon

    self.mpDraw = mp.solutions.drawing_utils 
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                 smooth_landmarks=self.smooth,
                                 min_detection_confidence=self.detectionCon,
                                 min_tracking_confidence=self.trackCon)
    
    #Set the dimensions of the canvas
    self.frameWidth = 640
    self.frameHeight = 480

  def findPose(self, img, draw=True):
    """
    findPose(img,draw)

    Takes an image, draws the landmarks for the pose estimation and the
    connections between them. Basically, the landmarks are the dots/joints, 
    and the connections are the lines/bones between landmarks. 

    Bones and joints are a analogy but not a technically precise one.
    """
    img = cv2.resize(img, (self.frameWidth, self.frameHeight))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)  #pose detected
    if self.results.pose_landmarks:
      if draw:
        self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, 
          self.mpPose.POSE_CONNECTIONS)
    return img

  def findPosition(self, img, draw=True):
    """
    findPosition(img, draw)

    Takes an image and returns the landmark points.
    """
    lmList = []
    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
        h,w,c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])
        if draw:
          cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
    return lmList

  
def main():
  #loop through camera feeds and show pose detection.
  cap = cv2.VideoCapture('../videos/pexels-anastasia-shuraeva-8751124.mp4')
  prevTime = 0
  detector = poseDetector()
  while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    #for this test function, I have chosen landmark 14, which is right elbow.
    cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), 
      cv2.FILLED)
    currentTime = time.time()
    fps = 1/(currentTime - prevTime)
    prevTime = currentTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, 
      (255,0,0),3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break;

if __name__ == "__main__":
  main()

#EOF
