import cv2
import numpy as np

# Load the image
image = cv2.imread('images/mock_gate.jpg')
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for RED
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Define HSV range for BLACK
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])  # Low brightness for black

# Define HSV range for WHITE
lower_white = np.array([0, 0, 200])     # High brightness, low saturation
upper_white = np.array([180, 30, 255])  # Slight tolerance for variations

# Create masks for red, black, and white colors
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_black = cv2.inRange(hsv, lower_black, upper_black)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Combine all masks into a single mask
mask_combined = cv2.bitwise_or(mask_red1, mask_red2)
mask_combined = cv2.bitwise_or(mask_combined, mask_black)
mask_combined = cv2.bitwise_or(mask_combined, mask_white)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(mask_combined, (5, 5), 0)

# Apply Morphological Transformations (Dilation) to connect parts of the gate
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(blurred, kernel, iterations=2)

# Find contours from the combined mask
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the gate)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Draw a bounding box around the entire gate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Gate', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
