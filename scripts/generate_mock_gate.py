import cv2
import numpy as np

# Create a blank light blue image
image = np.ones((600, 800, 3), dtype=np.uint8) * 255
image[:] = (255, 204, 153)  # Light blue background (BGR format)

# Draw the gate frame (PVC-like structure in white), only top and sides
frame_thickness = 10  # Reverted back to original thickness

# Define gate dimensions (2:1 ratio)
gate_width = 400  # 2 times the height
gate_height = 200

# Gate position (centered horizontally)
x_start = (800 - gate_width) // 2
x_end = x_start + gate_width
y_start = 200
y_end = y_start + gate_height

# Draw top bar
cv2.rectangle(image, (x_start, y_start), (x_end, y_start + frame_thickness), (255, 255, 255), -1)

# Draw left side bar
cv2.rectangle(image, (x_start, y_start), (x_start + frame_thickness, y_end), (255, 255, 255), -1)

# Draw right side bar
cv2.rectangle(image, (x_end - frame_thickness, y_start), (x_end, y_end), (255, 255, 255), -1)

# Draw the black box on the top-left
cv2.rectangle(image, (x_start - 10, y_start + 30), (x_start + 20, y_end - 100), (0, 0, 0), -1)  

# Draw the red box on the botom-left
cv2.rectangle(image, (x_start - 10, y_start + 100), (x_start + 20, y_end - 10), (0, 0, 255), -1)  

# Draw the red box on the top-right
cv2.rectangle(image, (x_end - 20, y_start + 30), (x_end + 10 , y_start + 120), (0, 0, 255), -1)  

# Draw the red box on the top-right
cv2.rectangle(image, (x_end - 20, y_start + 100), (x_end + 10 , y_end - 10), (0, 0, 0), -1)  

# Draw the red divider in the middle (half the height)
cv2.rectangle(image, ((x_start + x_end) // 2 - 5, y_start), ((x_start + x_end) // 2 + 5, y_start + gate_height // 2), (0, 0, 255), -1)  

# Display the mock-up image
cv2.imshow('Mock Gate', image)
cv2.imwrite('images/mock_gate.jpg', image)  # Save the mock image for testing
cv2.waitKey(0)
cv2.destroyAllWindows()