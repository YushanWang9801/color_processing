import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()

    gray = cv2.cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)
    segmented_selfie = cv2.bitwise_and(frame, mask)

    cv2.imshow("imgStacked", edges)
    if cv2.waitKey(1) == ord('q'):  # quit if the 'q' key is pressed
        break

cam.release()  # release the camera
cv2.destroyAllWindows()  # close all windows