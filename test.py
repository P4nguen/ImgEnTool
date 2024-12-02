import cv2

# Load an image
image = cv2.imread('s2.jpg', cv2.IMREAD_GRAYSCALE)

# Create a CLAHE object (optional parameters control the size of tiles and clipping limit)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE to the grayscale image
enhanced_image = clahe.apply(image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
