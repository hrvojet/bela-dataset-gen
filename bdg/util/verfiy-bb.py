import cv2
import os

image_path = "output/images/train/img_000042.jpg"
label_path = "output/images/train/img_000042.txt"

img = cv2.imread(image_path)
h, w = img.shape[:2]

with open(label_path) as f:
    lines = f.readlines()

for line in lines:
    cls, cx, cy, bw, bh = map(float, line.split())

    cx *= w
    cy *= h
    bw *= w
    bh *= h

    x1 = int(cx - bw/2)
    y1 = int(cy - bh/2)
    x2 = int(cx + bw/2)
    y2 = int(cy + bh/2)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("check", img)
cv2.waitKey(0)