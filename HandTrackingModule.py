import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode = False , maxHands = 2 , detectionCon = 0.5 , trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands , self.detectionCon ,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img , draw = True ):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:  # Check hand is in frame or not 
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLms,self.mpHands.HAND_CONNECTIONS) 
        return img
    
    def findposition(self , img , handNo = 0 , draw = True):
        lmList = []

        if self.result.multi_hand_landmarks :
            for handLms in self.result.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w) , int(lm.y*h)
                    # print("ID ",id," ",cx,cy)
                    lmList.append([id,cx,cy])
                    # if id == 0:
                    # if draw:
                        # cv2.circle(img, (cx,cy), 5 , (255,255,255) , cv2.FILLED)
        
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detactor = HandDetector()
    while(True):
        ret, img = cap.read()
        img = detactor.findHands(img) 
        lmList = detactor.findposition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)  #putting text on the frame 

        cv2.imshow("Image", img) #used to display an image in a window.
        cv2.waitKey(1) 


if __name__ == "__main__":
    main()
        