import cv2
import time
import os
import HandTrackingModule as htm

wCam , hCam = 640,480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
folderPath = "img"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    # print(f'{folderPath}/{imPath}')
# print(len(overlayList))
# print(overlayList)
pTime = 0

detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]
while True:
    ret , img = cap.read()
    img = detector.findHands(img,draw=False)
    lmList = detector.findposition(img,draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingersrig = []
        fingerslef = []
        if lmList[tipIds[1]][1] > lmList[tipIds[2]][1] :
            #print("RIGHT")
            # for thumbh
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingersrig.append(1)
            else:
                fingersrig.append(0)

            # for fingers
            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingersrig.append(1)
                else:
                    fingersrig.append(0)
            # print(fingers)    
            
        
        if lmList[tipIds[1]][1] < lmList[tipIds[2]][1]:
            #print("LEFT")
            # For thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingerslef.append(1)
            else:
                fingerslef.append(0)
            
            #for fingers
            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingerslef.append(1)
                else:
                    fingerslef.append(0)
        
        #print("LEN OF FINGERS Ri",fingersrig.count(1))
        #print("LEN OF FINGERS le",fingerslef.count(1))
        totalFingers = fingerslef.count(1) + fingersrig.count(1)
        #print("TOTAL FINgers is :  ",totalFingers)

        h,w,c = overlayList[totalFingers-1].shape
        img[0:h , 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0) , cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img, f'FPS : {int(fps)}',(210,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)  #putting text on the frame 
    img = cv2.resize(img,(1024,768))
    cv2.imshow("Image",img)
    cv2.waitKey(10)
    if cv2.waitKey(2) == 27:
        break
# cv2.destroyAllWindows()
