import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math

cap = cv2.VideoCapture(0)
detector = HandDetector()

resize = 20
img_size = 300

folder = "data/0"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    image = img
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
        imgcrop = img[y - resize:y + h + resize, x - resize:x + w + resize]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = img_size / h
            wcal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop, (wcal, img_size))
            imgResizeShape = imgResize.shape
            wgal = math.ceil((img_size - wcal) / 2)
            img_white[:, wgal:wcal + wgal] = imgResize
            cv2.imshow("hand", img_white)


    cv2.imshow("input", image)

    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', img_white)
    if key == ord("q"):
        break
