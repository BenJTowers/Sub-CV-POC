import cv2
import numpy as np

# Load the image
image = cv2.imread('images/mock_gate2.jpg')
if image is None:
    print("Error: Image not found.")
    exit()

# ===== Step 1: Apply CLAHE (Adaptive Histogram Equalization) =====
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l = clahe.apply(l)

lab = cv2.merge((l, a, b))
image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ===== Step 2: Define Updated HSV Ranges for Underwater Detection =====

# **Red (Adjusted for Water Distortion)**
lower_red1 = np.array([0, 100, 60])    
upper_red1 = np.array([10, 255, 255])  
lower_red2 = np.array([170, 100, 60])  
upper_red2 = np.array([180, 255, 255])  

# **Black (Ensure Darker Areas Are Captured)**
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 200, 55])  

# **White (Prevent Reflection Overdetection)**
lower_white = np.array([0, 0, 170])    
upper_white = np.array([180, 30, 255])  

# **Reject Greenish Pool Colors (Reflections)**
lower_reject_green = np.array([35, 50, 50])  
upper_reject_green = np.array([90, 255, 255])  

# ===== Step 3: Create & Combine Masks =====
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_black = cv2.inRange(hsv, lower_black, upper_black)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
mask_reject_green = cv2.inRange(hsv, lower_reject_green, upper_reject_green)  

# Combine masks & remove reflections
mask_combined = cv2.bitwise_or(mask_red1, mask_red2)
mask_combined = cv2.bitwise_or(mask_combined, mask_black)
mask_combined = cv2.bitwise_or(mask_combined, mask_white)
mask_combined = cv2.bitwise_and(mask_combined, cv2.bitwise_not(mask_reject_green))  

# ===== Step 4: Use Bilateral Filtering to Reduce Noise While Preserving Edges =====
filtered = cv2.bilateralFilter(mask_combined, d=9, sigmaColor=100, sigmaSpace=100)

# ===== Step 5: Apply Morphological Transformations =====
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(filtered, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# ===== Step 6: Apply Edge Detection (Canny) to Refine Detection =====
edges = cv2.Canny(closed, 50, 150)  # Detect edges to remove water reflections

# Combine edge detection with existing mask
final_mask = cv2.bitwise_and(closed, edges)

# ===== Step 7: Find and Filter Contours =====
contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Keep only reasonable-sized contours (removes false large reflections)
valid_contours = [cnt for cnt in contours if 1000 < cv2.contourArea(cnt) < 50000]

if valid_contours:
    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Draw a bounding box around the refined detected gate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the corrected image
    cv2.imwrite('output/detected_gate.jpg', image)
    print("Updated detection saved as 'output/detected_gate.jpg'")

# ===== Step 8: Display the Result =====
cv2.imshow('Refined Detected Gate', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
