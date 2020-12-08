import numpy as np
import cv2
import screeninfo


window_name = 'projector'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name,-3840, 0)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# image = np.ones((500, 500, 3), dtype=np.float32)
image = np.zeros((500, 500, 3), dtype=np.float32)
cv2.imshow(window_name, image)
cv2.waitKey()
cv2.destroyAllWindows()