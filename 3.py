#
#
# import cv2
# import numpy as np
# import os
# import pandas as pd
# from datetime import datetime
#
# # Khởi tạo bộ nhận diện khuôn mặt
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer/trainer.yml')  # Đảm bảo file trainer.yml tồn tại
# cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Sử dụng đường dẫn chuẩn của OpenCV
# faceCascade = cv2.CascadeClassifier(cascadePath)
#
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# # Khởi tạo id counter
# id = 0
#
# # Tên liên quan đến id: ví dụ ==> Marcelo: id=1, etc
# names = ['None', 'lai', 'bác']
#
# # Khởi tạo và bắt đầu video capture
# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set video width
# cam.set(4, 480)  # set video height
#
# # Định nghĩa kích thước cửa sổ tối thiểu để nhận diện khuôn mặt
# minW = 0.1 * cam.get(3)
# minH = 0.1 * cam.get(4)
#
# # Tạo DataFrame để lưu kết quả
# columns = ['ID', 'Name', 'Confidence', 'Timestamp']
# df = pd.DataFrame(columns=columns)
#
# while True:
#     ret, img = cam.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(int(minW), int(minH)),
#     )
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#
#         # Kiểm tra confidence, nếu nhỏ hơn 100 thì là kết quả tốt
#         if confidence < 100:
#             name = names[id]
#             confidence_percent = round(100 - confidence)
#         else:
#             name = "unknown"
#             confidence_percent = round(100 - confidence)
#
#         cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#         cv2.putText(img, str(confidence_percent) + "%", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
#
#         # Lưu kết quả vào DataFrame
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         new_row = pd.DataFrame({'ID': [id], 'Name': [name], 'Confidence': [confidence_percent], 'Timestamp': [timestamp]})
#         df = pd.concat([df, new_row], ignore_index=True)
#
#     cv2.imshow('camera', img)
#
#     k = cv2.waitKey(10) & 0xff  # Nhấn 'ESC' để thoát
#     if k == 27:
#         break
#
# # Lưu DataFrame vào file Excel
# excel_filename = 'face_recognition_results.xlsx'
# df.to_excel(excel_filename, index=False)
# print(f"\n[INFO] Results saved to {excel_filename}")
#
# # Dọn dẹp
# print("\n[INFO] Exiting Program and cleanup stuff")
# cam.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Make sure this file exists
cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize id counter
id = 0

# Names related to ids: example ==> Marcelo: id=1, etc
names = ['None', 'lai', 'bác']  # Make sure your trainer.yml matches these IDs

# Initialize and start video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Create DataFrame to store results
columns = ['ID', 'Name', 'Confidence', 'Timestamp']
df = pd.DataFrame(columns=columns)

# Confidence threshold (adjust as needed)
CONFIDENCE_THRESHOLD = 70

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is good (lower is better)
        if confidence < 100:
            confidence_percent = round(100 - confidence)
            # Only accept predictions with sufficient confidence
            if confidence_percent > CONFIDENCE_THRESHOLD:
                # Check if ID is within the names list range
                if id >= 0 and id < len(names):
                    name = names[id]
                else:
                    name = "unknown"
            else:
                name = "unknown"
        else:
            name = "unknown"
            confidence_percent = round(100 - confidence)

        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, f"{confidence_percent}%", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Save result to DataFrame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame(
            {'ID': [id], 'Name': [name], 'Confidence': [confidence_percent], 'Timestamp': [timestamp]})
        df = pd.concat([df, new_row], ignore_index=True)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' to exit
    if k == 27:
        break

# Save DataFrame to Excel
excel_filename = 'face_recognition_results.xlsx'
df.to_excel(excel_filename, index=False)
print(f"\n[INFO] Results saved to {excel_filename}")

# Cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
