import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x = 0
prev_y = 0

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            h, w, c = img.shape

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas,
                     (prev_x, prev_y),
                     (x, y),
                     (255, 255, 255),
                     5)

            prev_x, prev_y = x, y

    img = cv2.add(img, canvas)

    cv2.imshow("Air Drawing", img)

    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
