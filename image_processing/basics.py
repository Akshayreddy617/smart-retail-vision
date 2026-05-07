import cv2

# Read image
image = cv2.imread("images/shelf.jpg")

# Check image
if image is None:
    print("Image not found")
    exit()

# Resize image
image = cv2.resize(image, (800, 500))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges
edges = cv2.Canny(blur, 50, 200)

# Show outputs
cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Edges", edges)

# Save edge image
cv2.imwrite("outputs/edges.jpg", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()