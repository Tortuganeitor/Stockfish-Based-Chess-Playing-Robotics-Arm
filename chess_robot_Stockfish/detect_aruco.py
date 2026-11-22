
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Recherche automatique de la camera
cap = None

for camera_id in [1, 2, 3, 4]:
    test = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if test.isOpened():
        ok, frame = test.read()
        if ok and frame is not None and frame.mean() > 5:
            cap = test
            print("Camera found:", camera_id)
            break
    test.release()

if cap is None:
    print("No working camera found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        detected = sorted(ids.flatten().tolist())

        cv2.putText(
            frame,
            f"Pieces detected: {len(detected)}/12",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        print("Detected IDs:", detected, end="\r")

    cv2.imshow("Chess Robot - ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
