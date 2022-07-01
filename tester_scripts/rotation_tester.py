import cv2

img = cv2.imread("test_imgs/test_imgs.001.jpeg")

rows = img.shape[0]
cols = img.shape[1]

img_center = (cols / 2, rows / 2)
M = cv2.getRotationMatrix2D(img_center, 45, 1)

rotated_image = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))

cv2.imshow("rot", rotated_image)
cv2.waitKey()