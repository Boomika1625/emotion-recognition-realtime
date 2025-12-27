
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from emotion_labels import EMOTION_LABELS

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("models/emotion_model.h5")

# -----------------------------
# Load Haar Cascade for face detection
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Resize to model input size
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Normalize
        roi = roi_gray / 255.0

        # Reshape for model
        roi = np.reshape(roi, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(roi, verbose=0)
        emotion_label = EMOTION_LABELS[np.argmax(prediction)]

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    # Show output
    cv2.imshow("Real-Time Emotion Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()
