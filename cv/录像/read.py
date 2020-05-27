import cv2

cap = cv2.VideoCapture('VideoRecord.avi')  # 打开相机

while (True):
    ret, frame = cap.read()  # 捕获一帧图像
    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
    else:
        break

cap.release()  # 关闭相机
cv2.destroyAllWindows()  # 关闭窗口