
import cv2
import os

# Create folder for face images
folder = "my_face"
os.makedirs(folder, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
max_images = 30

print("Show your face to the camera.")
print("Press S to save a face image.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not working!")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )

    cv2.putText(
        frame,
        f"Saved: {count}/{max_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) == 1 and count < max_images:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            filename = os.path.join(
                folder, f"face_{count}.jpg"
            )

            cv2.imwrite(filename, face)
            count += 1
            print("Saved:", filename)

        else:
            print("Keep exactly one face visible.")

    if count >= max_images:
        print("Registration completed!")
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
