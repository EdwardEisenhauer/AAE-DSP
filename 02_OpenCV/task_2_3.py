# Task 2.3. Face tracking/detection:
# - Install necessary libraries.
# - Run script vidFace.py to recognize AI generated faces and check how it works
#   on image you choose.
# - Modify script vidFace.py (real time face tracking) to show recognized face
#   coordinates on image you pick up.
import sys

import cv2

from utils import suffix_filename


if __name__ == "__main__":
    filename: str = sys.argv[1]

    img = cv2.imread(filename)
    assert img is not None

    img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(suffix_filename(filename, "_grayscale"), img_grayscale)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Detect the faces
    faces = face_cascade.detectMultiScale(img_grayscale, 1.1, 4)
    # Draw the rectangle around each face
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(suffix_filename(filename, "_face_detection"), img)

    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_thickness = 1
    red = (0, 0, 255)
    white = (255, 255, 255)
    background_width = 70
    background_height = 11
    for x, y, w, h in faces:
        label_text = f"{x + w // 2},{y + h // 2}"
        background_size = cv2.getTextSize(
            label_text, font_face, font_scale, font_thickness
        )
        background_width, background_height = background_size[0]
        # One more pixel for legibility
        background_height = background_height + 1
        cv2.rectangle(
            img,
            (x, y),
            (x + background_width, y - background_height),
            white,
            -1,
        )
        cv2.putText(
            img,
            f"{x},{y}",
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            red,
        )

    cv2.imwrite(suffix_filename(filename, "_face_detection_coordinates"), img)
