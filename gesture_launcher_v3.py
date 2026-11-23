import cv2
import mediapipe as mp
import subprocess
import webbrowser

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()

# Gesture settings
stable_frames = 0
candidate_gesture = None
armed = True
status = "Show your fingers"
app_names = {
    1: "Notepad",
    2: "Calculator",
    3: "Google",
    4: "Paint"
}

# Voice feedback using Windows Speech
def speak(message):
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{message}'); $s.Dispose()"
    )

    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    except Exception as e:
        print("Voice Error:", e)


# Open selected application
def open_app(count):

    if count == 1:
        subprocess.Popen("notepad.exe")

    elif count == 2:
        subprocess.Popen("calc.exe")

    elif count == 3:
        webbrowser.open("https://www.google.com")

    elif count == 4:
        subprocess.Popen("mspaint.exe")

    print(app_names[count] + " Opened")
    speak(app_names[count] + " Opened")


try:

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            count = None

            if results.multi_hand_landmarks:

                for hand in results.multi_hand_landmarks:

                    lm = hand.landmark

                    # Count four fingers (thumb excluded)
                    fingers = [
                        lm[8].y < lm[6].y,
                        lm[12].y < lm[10].y,
                        lm[16].y < lm[14].y,
                        lm[20].y < lm[18].y
                    ]

                    count = sum(fingers)

                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

            # Gesture recognition
            if count == 0:

                stable_frames = 0
                candidate_gesture = None
                armed = True
                status = "READY - Show 1 to 4 fingers"

            elif count in [1, 2, 3, 4]:

                if not armed:
                    status = "Show closed fist to RESET"

                else:

                    if candidate_gesture == count:
                        stable_frames += 1

                    else:
                        candidate_gesture = count
                        stable_frames = 1

                    status = (
                        f"Detecting {app_names[count]}: "
                        f"{stable_frames}/20"
                    )

                    if stable_frames >= 20:

                        open_app(count)

                        status = (
                            f"{app_names[count]} OPENED"
                        )

                        armed = False
                        stable_frames = 0

            # Display finger count
            display_count = count if count is not None else 0

            cv2.putText(
                frame,
                f"Fingers: {display_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display status
            cv2.putText(
                frame,
                status,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to Exit",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "AI Gesture Launcher - Advanced 2.0",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

print("Gesture System Stopped")