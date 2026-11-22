
import cv2
import numpy as np

# =========================================
# CHESS ROBOT 4x4 - RED CORNER DETECTION
# =========================================

CAMERA_ID = 1
BOARD_SIZE = 4
WARP_SIZE = 600

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("USB Camera not detected!")

# HSV red color ranges
lower_red1 = np.array([0, 90, 70])
upper_red1 = np.array([12, 255, 255])

lower_red2 = np.array([170, 90, 70])
upper_red2 = np.array([179, 255, 255])

kernel = np.ones((5, 5), np.uint8)

def order_points(points):
    # Order: top-left, top-right,
    # bottom-right, bottom-left
    points = np.array(points, dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()

    return np.array([
        points[np.argmin(s)],
        points[np.argmin(diff)],
        points[np.argmax(s)],
        points[np.argmax(diff)]
    ], dtype=np.float32)

print("Camera connected!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red objects
    mask1 = cv2.inRange(
        hsv, lower_red1, upper_red1
    )

    mask2 = cv2.inRange(
        hsv, lower_red2, upper_red2
    )

    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Keep red objects with enough area
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 200:
            M = cv2.moments(contour)

            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                candidates.append((area, (cx, cy)))

    # Select four largest red objects
    candidates.sort(reverse=True)

    if len(candidates) >= 4:

        points = [
            item[1] for item in candidates[:4]
        ]

        corners = order_points(points)

        # Draw detected corners
        for i, point in enumerate(corners):
            x, y = point.astype(int)

            cv2.circle(
                frame, (x, y),
                8, (0, 255, 0), -1
            )

            cv2.putText(
                frame,
                f"P{i}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        destination = np.array([
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(
            corners, destination
        )

        warped = cv2.warpPerspective(
            frame,
            matrix,
            (WARP_SIZE, WARP_SIZE)
        )

        cell_size = WARP_SIZE // BOARD_SIZE

        # Draw 4x4 grid
        for i in range(1, BOARD_SIZE):
            pos = i * cell_size

            cv2.line(
                warped,
                (pos, 0),
                (pos, WARP_SIZE),
                (0, 255, 0),
                2
            )

            cv2.line(
                warped,
                (0, pos),
                (WARP_SIZE, pos),
                (0, 255, 0),
                2
            )

        # Add chess square labels
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                square = (
                    "ABCD"[col] +
                    str(4 - row)
                )

                x = col * cell_size + 15
                y = row * cell_size + 35

                cv2.putText(
                    warped,
                    square,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        cv2.imshow(
            "Chessboard 4x4 - Top View",
            warped
        )

    else:
        cv2.putText(
            frame,
            "Waiting for 4 red corners...",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Chess Robot - Red Corner Detection",
        frame
    )

    cv2.imshow(
        "Red Detection Mask",
        mask
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
