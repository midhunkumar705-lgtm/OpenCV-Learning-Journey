import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    totalFingers = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            fingers = []

            if lmList[tipIds[0]][0] > lmList[tipIds[0]-1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmList[tipIds[id]][1] < lmList[tipIds[id]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = sum(fingers)

    cv2.putText(img, str(totalFingers),
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                5)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
