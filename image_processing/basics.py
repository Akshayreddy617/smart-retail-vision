import cv2

# Read image
image = cv2.imread("images/shelf.jpg")

# Check image
if image is None:
    print("Image not found")
    exit()

# Resize
image = cv2.resize(image, (800, 500))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Print contour count
print("Contours found:", len(contours))

# Save output
cv2.imwrite("outputs/contours.jpg", image)

# Show outputs
cv2.imshow("Edges", edges)
cv2.imshow("Contours", image)

cv2.waitKey(0)
cv2.destroyAllWindows()