import cv2

name = input("Name: ")
start, end = int(input("Start: ")), int(input("End: "))
scale_percent = int(input("Scale percents: "))

for i in range(start, end):
    # read
    iname = name + str(i) + ".jpg"
    img = cv2.imread(iname, cv2.IMREAD_UNCHANGED)

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(iname, resized)

