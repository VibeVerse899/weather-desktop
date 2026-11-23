
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Camera start
cap = cv2.VideoCapture(0)

canvas = None
previous_x, previous_y = 0, 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera not found!")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                lm = hand.landmark

                # Index finger position
                x = int(lm[8].x * w)
                y = int(lm[8].y * h)

                # Finger detection
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                # Index finger only = Drawing
                if index_up and not middle_up:

                    if previous_x != 0:
                        cv2.line(
                            canvas,
                            (previous_x, previous_y),
                            (x, y),
                            (0, 255, 0),
                            5
                        )

                    previous_x, previous_y = x, y

                else:
                    # Stop drawing
                    previous_x, previous_y = 0, 0

                # Show fingertip
                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    (0, 0, 255),
                    -1
                )

        else:
            previous_x, previous_y = 0, 0

        # Combine camera and drawing
        output = cv2.addWeighted(
            frame, 1,
            canvas, 1,
            0
        )

        cv2.putText(
            output,
            "INDEX: DRAW | TWO FINGERS: MOVE",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("AI Air Drawing", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            canvas[:] = 0

        elif key == ord('s'):
            cv2.imwrite("my_drawing.png", canvas)
            print("Drawing Saved!")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Air Drawing Stopped!")
