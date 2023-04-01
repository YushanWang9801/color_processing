import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()

    cv2.imshow("imgStacked", frame)
    if cv2.waitKey(1) == ord('q'):  # quit if the 'q' key is pressed
        break

cam.release()  # release the camera
cv2.destroyAllWindows()  # close all windows