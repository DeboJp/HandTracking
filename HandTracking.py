import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1, 0.5,0.5)  # change parameters as needed.
mpDraw = mp.solutions.drawing_utils

# Fps initialization
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # reading RGB image and processing RGB frames
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) #check if tracking hands.

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):  # Grabbing relevant information of finger id landmarks
                # print(id, lm)
                h, w, c = img.shape  # Converting ratios to screen size
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)  # Cleaned up numbers
                # if id == 4: # Checking id Detection
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # draws hands and the connecting lines.

    # Fps Updating
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)  # Prints FPS

    cv2.imshow("Image", img)
    cv2.waitKey(1)
