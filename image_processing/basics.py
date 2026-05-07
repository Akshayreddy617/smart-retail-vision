import cv2

# Read image
image = cv2.imread("images/shelf.jpg")

if image is None:
    print("Image not found")
    exit()

# Resize image
image = cv2.resize(image, (800, 500))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Loop through contours
for contour in contours:

    # Ignore tiny contours
    area = cv2.contourArea(contour)

    if area > 500:

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

# Show result
cv2.imshow("Bounding Boxes", image)

# Save output
cv2.imwrite("outputs/bounding_boxes.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()