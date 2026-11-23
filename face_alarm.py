import numpy as np
import cv2
import os
import time
import winsound

# Face images folder
DATA_FOLDER = "my_face"

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Check face recognition module
if not hasattr(cv2, "face"):
    raise RuntimeError(
        "OpenCV contrib install check required"
    )

# Create face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

# Load registered face images
for filename in os.listdir(DATA_FOLDER):

    path = os.path.join(DATA_FOLDER, filename)

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(gray, (200, 200))

        faces.append(gray)
        labels.append(0)

if len(faces) < 2:
    raise RuntimeError(
        "Not enough registered face images!"
    )

# Train recognizer
labels = np.array(labels, dtype=np.int32)

labels = np.array(labels, dtype=np.int32)

labels = np.array(labels, dtype=np.int32)

recognizer.train(faces, labels)

print("Face Recognition Model Ready!")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not available")

no_face_start = None
last_alarm = 0

def alarm():
    global last_alarm

    if time.time() - last_alarm >= 2:
        winsound.Beep(1000, 500)
        last_alarm = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    unknown_found = False
    known_found = False

    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if label == 0 and confidence < 65:

            known_found = True

            color = (0, 255, 0)
            text = "MY FACE"

        else:

            unknown_found = True

            color = (0, 0, 255)
            text = "UNKNOWN FACE"

        cv2.rectangle(
            frame, (x, y),
            (x+w, y+h), color, 2
        )

        cv2.putText(
            frame, text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )

    # Unknown face alarm
    if unknown_found:

        no_face_start = None
        alarm()

        status = "WARNING: UNKNOWN FACE"

    # No face detected
    elif len(detected_faces) == 0:

        if no_face_start is None:
            no_face_start = time.time()

        if time.time() - no_face_start >= 2:
            alarm()

        status = "WARNING: FACE MISSING"

    else:

        no_face_start = None
        status = "MY FACE VERIFIED"

    cv2.putText(
        frame, status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.imshow("Face Security System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Security System Stopped")
