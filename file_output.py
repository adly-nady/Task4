import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
import math
import keras

cap = cv2.VideoCapture(0)
detector = HandDetector()

resize = 20
img_size = 300

index = 0

classifier = Classifier("model/keras_Model.h5", "model/labels.txt")

lables = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'U', 'V',
          'W', 'X', 'Y']
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
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
            per, index = classifier.getPrediction(img_white)

        cv2.putText(
            img=img,
            text=lables[index],
            org=(0, 100),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=3.0,
            color=(255, 55, 55),
            thickness=3
        )

    cv2.imshow("input", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
