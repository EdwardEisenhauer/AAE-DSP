# Task 2.2
#
# Install Python OpenCV libraries on yours system. Check if it works properly , then:
# - Load and display an image you pick up.
# - Convert to grayscale and apply blur (play with blur).
# - Apply edge detection.

import sys

import cv2

filename : str = sys.argv[1]

img = cv2.imread(filename)
assert img is not None

cv2.imwrite(filename, img)

# Task 2.3. Face tracking/detection:
# - Install necessary libraries.
# - Run script vidFace.py to recognize AI generated faces and check how it works
#   on image you choose.
# - Modify script vidFace.py (real time face tracking) to show recognized face
#   coordinates on image you pick up.
# Tas 2.4. Plates recognition:
# - Play with car.py script. Run it for different car images and try to tune to
#   recognize plate numbers properly.
# - Modify it to show each step of image process. Try to use OCR for automatic
#   plate recognition.
