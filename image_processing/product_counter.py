from ultralytics import YOLO
import cv2
from collections import Counter

# Load YOLO model
model = YOLO("yolov8n.pt")

# Read image
image = cv2.imread("images/shelf.jpg")

# Resize image
image = cv2.resize(image, (800, 500))

# Run detection
results = model(image)

# Get annotated frame
annotated_frame = results[0].plot()

# Get detected class IDs
class_ids = results[0].boxes.cls.tolist()

# Get class names
names = model.names

# Convert IDs to labels
labels = [names[int(class_id)] for class_id in class_ids]

# Count objects
object_counts = Counter(labels)

# Print counts
print("\nDetected Objects:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")

# Show image
cv2.imshow("Product Counter", annotated_frame)

# Save output
cv2.imwrite("outputs/product_counter.jpg", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()