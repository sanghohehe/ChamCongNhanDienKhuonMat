import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Không mở được webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Không đọc được hình ảnh từ webcam.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
