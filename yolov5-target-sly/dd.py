import cv2
import hole_detect

cap = cv2.VideoCapture(0)
a = hole_detect.yolo_detector()
path = 'upload/1/photo/2.jpg'
img = cv2.imread(path)
results = a.run(img)
if results:
    for i, pts in enumerate(results):
        cv2.rectangle(img, pts[0], pts[1], (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
