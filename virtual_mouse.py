
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Safety: move mouse to top-left corner to stop
pyautogui.FAILSAFE = True

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# Mouse smoothing
smooth = 5
prev_x, prev_y = 0, 0

# Click control
clicking = False
last_click = 0
click_delay = 0.5

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    try:
        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                # Index fingertip
                index_x = int(lm[8].x * width)
                index_y = int(lm[8].y * height)

                # Thumb tip
                thumb_x = int(lm[4].x * width)
                thumb_y = int(lm[4].y * height)

                # Check index finger up
                index_up = lm[8].y < lm[6].y

                if index_up:

                    # Map camera position to screen
                    target_x = int(
                        lm[8].x * screen_width
                    )

                    target_y = int(
                        lm[8].y * screen_height
                    )

                    # Smooth movement
                    curr_x = prev_x + (
                        target_x - prev_x
                    ) / smooth

                    curr_y = prev_y + (
                        target_y - prev_y
                    ) / smooth

                    pyautogui.moveTo(
                        int(curr_x),
                        int(curr_y)
                    )

                    prev_x = curr_x
                    prev_y = curr_y

                # Thumb and index distance
                distance = math.hypot(
                    index_x - thumb_x,
                    index_y - thumb_y
                )

                # Pinch = Left Click
                if distance < 35 and index_up:

                    if (
                        not clicking
                        and time.time() - last_click > click_delay
                    ):
                        pyautogui.click()
                        last_click = time.time()
                        clicking = True

                    status = "CLICK"

                else:
                    clicking = False
                    status = "MOVE MOUSE"

                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    10,
                    (0, 255, 255),
                    -1
                )

            else:
                clicking = False

            cv2.imshow("AI Virtual Mouse", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    except pyautogui.FailSafeException:
        print("Emergency stop activated!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Virtual Mouse Stopped!")
