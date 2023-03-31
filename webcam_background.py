import cv2
import numpy as np

import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cam = cv2.VideoCapture(0)
segmentor=SelfiSegmentation()
fpsReader=cvzone.FPS()
cam.set(3,480)
cam.set(4,684)

bg_img = cv2.imread("./images/bg1.jpg")
bg_img = cv2.resize(bg_img, (640,480))

while True:
    ret, frame = cam.read()
    # frame = cv2.flip(frame, 1)
    imgout=segmentor.removeBG(frame, bg_img,threshold=0.8)
    cv2.imshow("imgStacked",imgout)
    if cv2.waitKey(1) == ord('q'):  # quit if the 'q' key is pressed
        break

cam.release()  # release the camera
cv2.destroyAllWindows()  # close all windows