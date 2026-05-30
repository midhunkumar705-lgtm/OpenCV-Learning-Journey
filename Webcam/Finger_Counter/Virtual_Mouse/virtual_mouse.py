import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            mpDraw.draw_landmarks(
                img,
                hand,
                mpHands.HAND_CONNECTIONS
            )

            landmarks = hand.landmark

            index_finger = landmarks[8]

            x = int(index_finger.x * screen_w)
            y = int(index_finger.y * screen_h)

            pyautogui.moveTo(x, y)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
