import cv2
import os

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Khởi tạo camera
cam = cv2.VideoCapture(0)  # nếu không lên, thử đổi lại thành 0
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Kiểm tra camera có hoạt động không
if not cam.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

# Tải bộ phân loại khuôn mặt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Nhập ID khuôn mặt
face_id = input('\nEnter user ID and press <return> ==>  ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")
count = 0

while True:
    ret, img = cam.read()
    if not ret or img is None:
        print("❌ Không thể đọc ảnh từ webcam. Thử lại...")
        continue

    img = cv2.flip(img, 1)  # Lật ảnh để dễ nhìn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Lưu ảnh khuôn mặt
        save_path = f"dataset/User.{face_id}.{count}.jpg"
        print(f"Saving image to: {save_path}")
        try:
            cv2.imwrite(save_path, gray[y:y + h, x:x + w])
        except Exception as e:
            print(f"Error saving image: {e}")

        cv2.imshow('image', img)

    # Nhấn ESC để thoát
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 200:
        break

# Dọn dẹp
print("\n[INFO] Exiting program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
