import cv2
import numpy as np
from constant import lms_matrix, rgb_matrix
from constant import pro_sim_matrix, deu_sim_matrix, tri_sim_matrix

dim = (350,250)
top_x, top_y = 10, 30

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    lms_frame = np.tensordot(frame, lms_matrix, axes=([2], [1]))
    pro_frame = np.tensordot(lms_frame, pro_sim_matrix, axes=([2], [1]))
    pro_frame = np.tensordot(pro_frame, rgb_matrix, axes=([2], [1])).astype(np.uint8)
    deu_frame = np.tensordot(lms_frame, deu_sim_matrix, axes=([2],[1]))
    deu_frame = np.tensordot(deu_frame, rgb_matrix, axes=([2], [1])).astype(np.uint8)
    tri_frame = np.tensordot(lms_frame, tri_sim_matrix, axes=([2],[1]))
    tri_frame = np.tensordot(tri_frame, rgb_matrix, axes=([2], [1])).astype(np.uint8)

    frame = np.concatenate((np.concatenate((frame, pro_frame), axis=0), 
                           np.concatenate((deu_frame, tri_frame), axis = 0)), 
                           axis =1).astype(np.uint8)

    frame = cv2.putText(frame, "Original Frame", (top_x, top_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Protanopia", (top_x+dim[0], top_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Deuteranopia", (top_x, top_y+dim[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Tritanopia", (top_x+dim[0], top_y+dim[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)  # display the frame in a window
    if cv2.waitKey(1) == ord('q'):  # quit if the 'q' key is pressed
        break

cam.release()  # release the camera
cv2.destroyAllWindows()  # close all windows