from ultralytics import YOLO
import cv2
from collections import Counter

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (800, 500))

    # Run YOLO detection
    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Get class IDs
    class_ids = results[0].boxes.cls.tolist()

    # Get labels
    names = model.names
    labels = [names[int(class_id)] for class_id in class_ids]

    # Count objects
    object_counts = Counter(labels)

    # Print counts
    y_position = 30

    for obj, count in object_counts.items():

        text = f"{obj}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y_position += 30

    # Show frame
    cv2.imshow("Live Retail Vision", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()