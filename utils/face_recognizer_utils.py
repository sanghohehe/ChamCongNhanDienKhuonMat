import cv2
import numpy as np
import os
from PIL import Image
import config

# Hàm tải bộ phát hiện khuôn mặt
def load_face_detector():
    """
    Tải bộ phát hiện khuôn mặt (ví dụ: Haar Cascade Classifier).
    Sử dụng đường dẫn từ config.FACE_DETECTOR_PATH.
    """
    face_detector_path = config.FACE_DETECTOR_PATH
    if not os.path.exists(face_detector_path):
        print(f"Lỗi: Không tìm thấy file Haar Cascade tại {face_detector_path}")
        # Thử đường dẫn mặc định của OpenCV nếu không tìm thấy
        fallback_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(fallback_path):
            face_detector_path = fallback_path
            print(f"Thử tải từ đường dẫn mặc định của OpenCV: {fallback_path}")
        else:
            raise FileNotFoundError(f"Không thể tải bộ phát hiện khuôn mặt. File '{config.FACE_DETECTOR_PATH}' và '{fallback_path}' không tồn tại.")
    
    detector = cv2.CascadeClassifier(face_detector_path)
    if detector.empty():
        raise IOError("Không thể tải bộ phát hiện khuôn mặt. Vui lòng kiểm tra file XML.")
    print("Đã tải bộ phát hiện khuôn mặt.")
    return detector

# Hàm tải model nhận diện khuôn mặt
def load_recognizer_model():
    """Tải mô hình nhận diện khuôn mặt LBPH."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(config.MODEL_FILE):
        try:
            recognizer.read(config.MODEL_FILE)
            print(f"Đã tải model từ {config.MODEL_FILE}")
        except cv2.error as e:
            print(f"Lỗi đọc model từ {config.MODEL_FILE}: {e}")
            # Nếu có lỗi, có thể do file bị hỏng, trả về model rỗng và yêu cầu train lại
            print("Model có thể bị hỏng. Vui lòng huấn luyện lại model.")
            return cv2.face.LBPHFaceRecognizer_create() # Trả về model mới
    else:
        print(f"File model nhận diện không tồn tại: {config.MODEL_FILE}. Vui lòng train model.")
        return cv2.face.LBPHFaceRecognizer_create() # Trả về model mới
    return recognizer

# Hàm tải ánh xạ ID
def load_id_mapping():
    """Tải ánh xạ ID từ file."""
    id_mapping = {}
    if os.path.exists(config.ID_MAPPING_FILE):
        with open(config.ID_MAPPING_FILE, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(':')
                if len(parts) == 2:
                    numeric_id, original_id_name = parts # Đổi tên biến để rõ nghĩa hơn
                    try:
                        id_mapping[int(numeric_id)] = original_id_name
                    except ValueError:
                        # Nếu numeric_id không phải số, có thể là lỗi định dạng, bỏ qua dòng này
                        print(f"Cảnh báo: Dòng '{line.strip()}' trong id_mapping.txt không đúng định dạng. Bỏ qua.")
    print(f"Đã tải ID Mapping: {id_mapping}")
    return id_mapping

# Hàm lưu ánh xạ ID
def save_id_mapping(id_map):
    """Lưu ánh xạ ID vào file."""
    os.makedirs(os.path.dirname(config.ID_MAPPING_FILE), exist_ok=True)
    with open(config.ID_MAPPING_FILE, 'w') as f:
        for numeric_id, original_id_name in id_map.items():
            f.write(f"{numeric_id}:{original_id_name}\n")
    print(f"Đã lưu ID mapping vào {config.ID_MAPPING_FILE}")

# Hàm lấy ảnh và nhãn từ dataset (đã điều chỉnh để đọc từ User.ID_Name.idx.jpg)
def get_images_and_labels(path, detector):
    """Thu thập ảnh khuôn mặt và ID từ thư mục dataset."""
    imagePaths = []
    # Duyệt qua các thư mục con trong dataset_path
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                imagePaths.append(os.path.join(root, file))

    faceSamples = []
    original_ids = [] # Lưu trữ ID dạng string (e.g., "NV001")

    for imagePath in imagePaths:
        try:
            filename = os.path.split(imagePath)[-1]
            parts = filename.split('.')
            if len(parts) >= 3 and parts[0] == "User":
                # Lấy phần ID_NAME từ filename (ví dụ: User.NV001_TenA.1.jpg -> NV001_TenA)
                original_id_name = parts[1] 
                # Lấy user_id từ original_id_name (ví dụ: NV001_TenA -> NV001)
                user_id_part = original_id_name.split('_')[0] 

                PIL_img = Image.open(imagePath).convert('L') # Convert to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    original_ids.append(user_id_part) # Thêm user_id vào danh sách

        except Exception as e:
            print(f"Lỗi khi xử lý {imagePath}: {str(e)}")

    print(f"Đã thu thập {len(faceSamples)} mẫu khuôn mặt với {len(original_ids)} ID từ dataset.")
    return faceSamples, original_ids

# Hàm ánh xạ ID gốc sang ID số cho training
def map_ids_for_training(original_ids_list):
    """Tạo ánh xạ từ ID gốc (string) sang ID số (int) cho quá trình training."""
    unique_original_ids = sorted(list(set(original_ids_list))) # Sắp xếp để đảm bảo thứ tự nhất quán
    id_mapping_for_training = {id_str: i for i, id_str in enumerate(unique_original_ids)} # ánh xạ string to int
    numeric_ids = np.array([id_mapping_for_training[id_str] for id_str in original_ids_list])

    return numeric_ids, id_mapping_for_training # Trả về mapping string to int để train

# Hàm lấy tên người dùng từ ID gốc
def get_name_for_id(original_id_str):
    """
    Lấy tên người dùng từ ID gốc (string) bằng cách đọc nội dung file User.OriginalID.name.txt.
    Hoặc trích xuất từ original_id_str nếu nó có dạng "ID_Name".
    """
    # Ưu tiên đọc từ file User.ID.name.txt
    name_file_path = os.path.join(config.DATASET_PATH, f"User.{original_id_str}.name.txt")
    if os.path.exists(name_file_path):
        try:
            with open(name_file_path, 'r', encoding='utf-8') as f:
                name = f.readline().strip()
                if name:
                    return name
        except Exception as e:
            print(f"Lỗi khi đọc file tên {name_file_path}: {e}")
    
    # Nếu không tìm thấy file hoặc file rỗng, thử trích xuất từ original_id_str (nếu có dạng ID_Name)
    if '_' in original_id_str:
        parts = original_id_str.split('_', 1)
        if len(parts) == 2:
            return parts[1] # Trả về phần tên sau dấu gạch dưới

    return "Không xác định"

# Hàm lưu tên người dùng
def save_name_for_id(user_id, name):
    """Lưu tên người dùng vào file."""
    # Đảm bảo thư mục dataset tồn tại trước khi tạo file .name.txt
    os.makedirs(config.DATASET_PATH, exist_ok=True) 
    name_file = f"{config.DATASET_PATH}/User.{user_id}.name.txt"
    with open(name_file, 'w', encoding='utf-8') as f:
        f.write(name)
    print(f"Đã lưu tên '{name}' cho ID '{user_id}' vào file.")

# Hàm thu thập dataset
def collect_dataset(user_id, user_name, detector):
    """
    Thu thập dữ liệu khuôn mặt cho người dùng mới.
    user_id: ID định danh của người dùng (ví dụ: "NV001")
    user_name: Tên người dùng (ví dụ: "Nguyễn Văn A")
    detector: Bộ phát hiện khuôn mặt đã tải
    """
    # Tạo tên thư mục và tiền tố file dựa trên user_id và user_name
    original_id_name_str = f"{user_id}_{user_name}" 
    user_folder_path = os.path.join(config.DATASET_PATH, original_id_name_str)
    os.makedirs(user_folder_path, exist_ok=True) # Tạo thư mục cho người dùng

    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        raise IOError("Không thể mở camera. Vui lòng kiểm tra kết nối camera.")

    sample_num = 0
    print(f"Bắt đầu thu thập ảnh cho {user_name} (ID: {user_id}). Vui lòng đưa khuôn mặt vào khung hình...")

    while True:
        ret, img = cam.read()
        if not ret:
            print("Lỗi đọc khung hình từ camera.")
            break
        
        img = cv2.flip(img, 1) # Lật ảnh theo chiều ngang
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample_num += 1
            # Lưu ảnh vào thư mục của người dùng với định dạng User.ID_Name.SốThứTự.jpg
            img_name = f"User.{original_id_name_str}.{sample_num}.jpg"
            cv2.imwrite(os.path.join(user_folder_path, img_name), gray[y:y + h, x:x + w])
            cv2.imshow('Thu thap du lieu', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27 or k == ord('q'): # Nhấn 'ESC' hoặc 'q' để thoát
            break
        elif sample_num >= config.NUM_IMAGES_TO_CAPTURE: # Thu thập đủ số lượng mẫu
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập {sample_num} mẫu cho {user_name} (ID: {user_id}).")
    save_name_for_id(user_id, user_name) # Lưu tên người dùng vào file .name.txt
    return sample_num

# Hàm huấn luyện model
def train_recognizer(detector):
    """Huấn luyện mô hình nhận diện khuôn mặt và lưu nó."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Lấy danh sách các thư mục con trong dataset_path (mỗi thư mục là một user)
    # Tên thư mục con có thể là "UserID_Name"
    
    # Điều chỉnh để get_images_and_labels duyệt qua cả dataset_path
    all_face_samples, all_user_ids = get_images_and_labels(config.DATASET_PATH, detector)

    if len(all_face_samples) == 0:
        raise ValueError("Không tìm thấy khuôn mặt nào trong thư mục dataset để huấn luyện.")

    # Ánh xạ các user_id (string) thành numeric_id (int) cho quá trình huấn luyện
    # và tạo id_mapping cuối cùng để lưu
    
    # Tạo mapping từ user_id (string) sang numeric_id (int)
    unique_user_ids = sorted(list(set(all_user_ids)))
    user_id_to_numeric_id = {user_id: i for i, user_id in enumerate(unique_user_ids)}
    
    numeric_labels = np.array([user_id_to_numeric_id[uid] for uid in all_user_ids])

    # Tạo id_mapping_final để lưu vào file: numeric_id -> original_id_name (ví dụ: 0:NV001_TenA)
    id_mapping_final = {}
    for user_id_str, numeric_id in user_id_to_numeric_id.items():
        # Cần tìm tên đầy đủ của người dùng để lưu vào id_mapping.txt
        # Có thể lấy từ file .name.txt hoặc từ tên thư mục dataset
        # Duyệt qua các thư mục trong DATASET_PATH
        full_name_found = False
        for folder_name in os.listdir(config.DATASET_PATH):
            if os.path.isdir(os.path.join(config.DATASET_PATH, folder_name)) and folder_name.startswith(user_id_str + '_'):
                id_mapping_final[numeric_id] = folder_name # Lưu dạng NumericID:UserID_Name
                full_name_found = True
                break
        if not full_name_found:
             # Fallback nếu không tìm thấy thư mục (ví dụ: UserID chưa có ảnh nhưng đã train)
            id_mapping_final[numeric_id] = user_id_str # Chỉ lưu UserID nếu không tìm thấy tên đầy đủ
    
    recognizer.train(all_face_samples, numeric_labels)
    
    os.makedirs(config.TRAINER_PATH, exist_ok=True) # Đảm bảo thư mục trainer tồn tại
    recognizer.write(config.MODEL_FILE)
    save_id_mapping(id_mapping_final) # Lưu mapping đã tạo

    print(f"Đã huấn luyện model và lưu vào {config.MODEL_FILE}")
    print(f"Đã lưu ID mapping cuối cùng: {id_mapping_final}")