#%%

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dst = cv2.dilate(binary, kernel)  # 膨胀
# cv2.imshow("dst1", dst)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.erode(dst, kernel)  # 腐蚀