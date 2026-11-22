
import cv2
import os

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

os.makedirs("markers", exist_ok=True)

# 4 markers for board calibration
# 6 markers for chess pieces
ids = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]

for marker_id in ids:
    marker = cv2.aruco.generateImageMarker(
        dictionary, marker_id, 400
    )

    marker = cv2.copyMakeBorder(
        marker, 40, 40, 40, 40,
        cv2.BORDER_CONSTANT,
        value=255
    )

    cv2.imwrite(
        f"markers/aruco_{marker_id}.png",
        marker
    )

print("All ArUco markers generated!")
