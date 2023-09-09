import cv2
import numpy as np


numpy_matrix = np.random.randint(0, 256, (3,3), dtype=np.uint8)
print(numpy_matrix)

opencv_image = cv2.cvtColor(numpy_matrix, cv2.COLOR_GRAY2BGR)
print(opencv_image)
cv2.imwrite('opencv_image.jpg', opencv_image)