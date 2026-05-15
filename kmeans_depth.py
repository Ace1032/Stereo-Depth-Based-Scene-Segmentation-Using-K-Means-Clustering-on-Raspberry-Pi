import cv2
import numpy as np

# Load disparity image
img = cv2.imread("disparity_test2.png", cv2.IMREAD_GRAYSCALE)

# Convert to float
Z = img.reshape((-1,1))
Z = np.float32(Z)

# K-means parameters
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply K-means
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to image
center = np.uint8(center)
res = center[label.flatten()]
result = res.reshape((img.shape))

# Save result
cv2.imwrite("kmeans_depth_k3.png", result)

print("K-means clustering done!")
