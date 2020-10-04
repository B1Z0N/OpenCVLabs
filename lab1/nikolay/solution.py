import cv2

# take a webcam photo and save it on the disk
capture = cv2.VideoCapture(0)
ret, image = capture.read()
image = cv2.cvtColor(image, 0)
cv2.imshow("My random image", image)
cv2.imwrite("image.jpg", image)

# load from disk
image = cv2.imread("image.jpg")

# add line and rectangle to grayed photo
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.rectangle(image, (100, 50), (50, 100), (50, 100, 100), 2)
cv2.line(image, (-50, 600), (30, 240), (255, 255, 134), 2)
cv2.imshow("My image loaded from disk", image)
cv2.imwrite("image.jpg", image)

# dismiss everything
capture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()