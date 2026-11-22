
import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera 1 not available")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received")
        break

    cv2.imshow("CAMERA TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
