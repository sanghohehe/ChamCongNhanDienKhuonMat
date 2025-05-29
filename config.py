import os
import cv2 

# Base directory for data (thư mục gốc của dự án )
# Đảm bảo BASE_DIR là thư mục chứa admin_app.py, main_app.py, và các thư mục con như database, utils, v.v.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Lấy thư mục chứa file config.py

# Paths for dataset and trainer
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer")

# Model file for face recognition
MODEL_FILE = os.path.join(TRAINER_PATH, "trainer.yml")

# ID mapping file
ID_MAPPING_FILE = os.path.join(TRAINER_PATH, "id_mapping.txt")

# Attendance CSV file
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv") 

# Face detection cascade classifier 
FACE_DETECTOR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Confidence threshold for recognition
CONFIDENCE_THRESHOLD = 50

# Đường dẫn đến các file âm thanh
SOUNDS_FOLDER = os.path.join(BASE_DIR, "sounds")
SUCCESS_SOUND = os.path.join(SOUNDS_FOLDER, "success.wav")
FAIL_SOUND = os.path.join(SOUNDS_FOLDER, "fail.wav")
UNKNOWN_SOUND = os.path.join(SOUNDS_FOLDER, "unknown.wav")
CHECKOUT_SOUND = os.path.join(SOUNDS_FOLDER, "checkout.mp3")

# Thời gian cooldown (giây) giữa các lần chấm công cho cùng một người
COOLDOWN_TIME = 10 

# Số lượng ảnh mẫu để thu thập cho mỗi người
NUM_IMAGES_TO_CAPTURE = 30 

CAMERA_INDEX = 0 # Chỉ số camera mặc định (thường là 0)

# Thời gian tối thiểu giữa hai lần check-in/check-out liên tiếp của cùng một người để tránh trùng lặp
MIN_CHECK_INTERVAL_SECONDS  = 0
MAX_WORK_SESSION_HOURS = 12 * 3600 # 12 giờ