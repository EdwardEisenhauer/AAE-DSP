# Task 2.2
#
# Install Python OpenCV libraries on yours system. Check if it works properly , then:
# - Load and display an image you pick up.
# - Convert to grayscale and apply blur (play with blur).
# - Apply edge detection.

import sys

import cv2

from utils import suffix_filename


def detect_edges(img):
    # Based on https://opencv.org/blog/edge-detection-using-opencv/
    # Apply Sobel operator
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    return gradient_magnitude


if __name__ == "__main__":
    filename: str = sys.argv[1]

    img = cv2.imread(filename)
    assert img is not None

    img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(suffix_filename(filename, "_grayscale"), img_grayscale)

    img_edge_detection = detect_edges(img_grayscale)
    cv2.imwrite(suffix_filename(filename, "_edge_detection"), img_edge_detection)

    for kernel_size in [1, 3, 5, 8, 13]:
        img_blur = cv2.blur(img, (kernel_size, kernel_size))
        cv2.imwrite(suffix_filename(filename, "_blur_" + str(kernel_size)), img_blur)

        img_edge_detection = detect_edges(img_blur)
        cv2.imwrite(
            suffix_filename(filename, "_edge_detection_blur_" + str(kernel_size)),
            img_edge_detection,
        )
