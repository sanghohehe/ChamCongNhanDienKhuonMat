# import cv2
# import numpy as np
# from PIL import Image
# import os
#
# # Đường dẫn thư mục chứa ảnh khuôn mặt
# path = 'dataset'
#
# # Tạo bộ nhận dạng khuôn mặt LBPH
# recognizer = cv2.face.LBPHFaceRecognizer_create()
#
# # Sử dụng đường dẫn chuẩn của OpenCV để tải file cascade
# cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# detector = cv2.CascadeClassifier(cascade_path)
#
# # Hàm lấy ảnh và ID từ dataset
# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     faceSamples = []
#     ids = []
#
#     for imagePath in imagePaths:
#         PIL_img = Image.open(imagePath).convert('L')  # chuyển về ảnh xám
#         img_numpy = np.array(PIL_img, 'uint8')
#         id = int(os.path.split(imagePath)[-1].split(".")[1])
#         faces = detector.detectMultiScale(img_numpy)
#
#         for (x, y, w, h) in faces:
#             faceSamples.append(img_numpy[y:y + h, x:x + w])
#             ids.append(id)
#
#     return faceSamples, ids
#
# # Huấn luyện
# print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
# faces, ids = getImagesAndLabels(path)
# recognizer.train(faces, np.array(ids))
#
# # Tạo thư mục trainer nếu chưa có
# if not os.path.exists('trainer'):
#     os.makedirs('trainer')
#
# # Lưu model
# recognizer.write('trainer/trainer.yml')
#
# print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


import cv2
import numpy as np
from PIL import Image
import os

# Path to face images
path = 'dataset'

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load cascade classifier
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"[ERROR] Cascade file not found at {cascade_path}")
    exit()

detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    print("[ERROR] Failed to load cascade classifier")
    exit()


def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    # Check if dataset path exists
    if not os.path.exists(path):
        print(f"[ERROR] Dataset path '{path}' not found")
        return faceSamples, ids

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

    if not imagePaths:
        print("[ERROR] No images found in dataset directory")
        return faceSamples, ids

    for imagePath in imagePaths:
        try:
            # Open and convert image
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract ID from filename (expected format: name.id.jpg)
            filename = os.path.split(imagePath)[-1]
            try:
                id = int(filename.split(".")[-2])  # gets the id part
            except (IndexError, ValueError):
                print(f"[WARNING] Skipping file with invalid name format: {filename}")
                continue

            # Detect faces
            faces = detector.detectMultiScale(img_numpy)

            if len(faces) == 0:
                print(f"[WARNING] No faces detected in {filename}")
                continue

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        except Exception as e:
            print(f"[ERROR] Processing {imagePath}: {str(e)}")
            continue

    return faceSamples, ids


print("\n[INFO] Training faces. It will take a few seconds. Wait...")
faces, ids = getImagesAndLabels(path)

if len(faces) == 0:
    print("\n[ERROR] No faces found in dataset. Please check:")
    print("1. Dataset path is correct")
    print("2. Images contain clear frontal faces")
    print("3. Image naming follows 'name.id.jpg' format")
    exit()

print(f"[INFO] Found {len(faces)} face samples for {len(np.unique(ids))} persons")

# Train the recognizer
recognizer.train(faces, np.array(ids))

# Create trainer directory if not exists
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Save the model
recognizer.write('trainer/trainer.yml')

print("\n[INFO] {0} faces trained. Model saved to trainer/trainer.yml".format(len(np.unique(ids))))