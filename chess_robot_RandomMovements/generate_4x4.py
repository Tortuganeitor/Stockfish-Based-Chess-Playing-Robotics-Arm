
import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker_size = 200
margin = 30
cell_size = marker_size + 2 * margin

sheet = np.full(
    (cell_size * 4, cell_size * 4),
    255,
    dtype=np.uint8
)

for marker_id in range(16):
    marker = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        marker_size
    )

    row = marker_id // 4
    col = marker_id % 4

    x = col * cell_size + margin
    y = row * cell_size + margin

    sheet[
        y:y + marker_size,
        x:x + marker_size
    ] = marker

cv2.imwrite("aruco_4x4.png", sheet)

print("SUCCESS: 16 ArUco markers generated!")
