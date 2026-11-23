
import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Colour list (BGR format)
colors = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255)   # Purple
]

color_index = 0
brush_size = 8
eraser_size = 35

cap = cv2.VideoCapture(0)

canvas = None
previous_point = None
previous_gesture = "none"

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Webcam not found!")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        if canvas is None:
            canvas = cv2.imread(
                "drawing_canvas.jpg"
            )

            if canvas is None or canvas.shape[:2] != (height, width):
                canvas = 255 * __import__("numpy").ones(
                    (height, width, 3), dtype="uint8"
                )

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        gesture = "none"
        current_point = None

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand.landmark

            # Finger detection
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            # Index finger position
            x = int(lm[8].x * width)
            y = int(lm[8].y * height)

            current_point = (x, y)

            # Gesture recognition
            if index_up and middle_up and ring_up and pinky_up:
                gesture = "eraser"

            elif index_up and middle_up:
                gesture = "color"

            elif index_up:
                gesture = "draw"

            else:
                gesture = "none"

            # Change colour once per gesture
            if gesture == "color" and previous_gesture != "color":
                color_index = (color_index + 1) % len(colors)

            # Drawing
            if gesture == "draw":

                if previous_point is not None:
                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        colors[color_index],
                        brush_size
                    )

                previous_point = current_point

            elif gesture == "eraser":

                if previous_point is not None:
                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        (255, 255, 255),
                        eraser_size
                    )

                previous_point = current_point

            else:
                previous_point = None

            # Show current fingertip
            cv2.circle(
                frame,
                current_point,
                10,
                (0, 255, 255),
                -1
            )

        else:
            previous_point = None

        previous_gesture = gesture

        # Show information
        cv2.putText(
            frame,
            "MODE: " + gesture.upper(),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Show selected colour
        cv2.rectangle(
            frame,
            (20, 60),
            (80, 100),
            colors[color_index],
            -1
        )

        # Display windows
        cv2.imshow("AI Hand Tracking", frame)
        cv2.imshow("AI Air Drawing Canvas", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            canvas[:] = 255

        elif key == ord('s'):
            cv2.imwrite("my_artwork.png", canvas)
            print("Artwork Saved!")

cap.release()
cv2.destroyAllWindows()

print("AI Air Drawing Stopped!")
