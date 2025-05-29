import cv2
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

from utils.camera_utils import initialize_camera, release_camera, load_face_detector
from utils.face_recognizer_utils import load_recognizer_model, load_id_mapping, get_name_for_id
import config
from database.database_manager import record_attendance

class MainFunctions:
    def __init__(self, root, video_label, status_label, result_label, names):
        self.root = root
        self.video_label = video_label
        self.status_label = status_label
        self.result_label = result_label
        self.names = names

        self.camera = None
        self.is_recognizing = False
        self.faceCascade = None # Khởi tạo là None
        self.recognizer = None
        self.id_mapping = {}

        self.last_recognized_time = {} # Sử dụng dictionary để theo dõi thời gian cuối cùng của mỗi ID
        self.recognition_cooldown_time = 10 # Giây, thời gian chờ trước khi chấm công lại cho cùng một người

    def _load_recognition_components(self):
        """Tải mô hình nhận diện, bộ phát hiện khuôn mặt và ánh xạ ID."""
        try:
            # Tải recognizer
            self.recognizer = load_recognizer_model()
            if not os.path.exists(config.MODEL_FILE):
                messagebox.showerror("Lỗi", "Không tìm thấy file model (trainer.yml). Vui lòng train model trước từ Admin Panel.")
                return False
            
            # Tải face detector
            self.faceCascade = load_face_detector() # Gọi từ utils.camera_utils
            
            # Tải ID mapping
            self.id_mapping = load_id_mapping()
            if not self.id_mapping:
                messagebox.showerror("Lỗi", "Không tìm thấy ánh xạ ID (id_mapping.txt). Vui lòng train model lại từ Admin Panel.")
                return False
            
            return True
        except (FileNotFoundError, IOError) as e:
            messagebox.showerror("Lỗi", f"Không thể tải model hoặc các thành phần: {str(e)}\n"
                                       "Hãy đảm bảo bạn đã train model và file 'haarcascade_frontalface_default.xml' tồn tại.")
            return False
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi tải các thành phần: {str(e)}")
            return False

    def start_recognition(self, check_type):
        """Bắt đầu quá trình nhận diện khuôn mặt và hiển thị lên camera."""
        if self.is_recognizing:
            self.status_label.config(text="Camera đang hoạt động...", foreground="orange")
            return

        # Tải các thành phần nhận diện trước khi mở camera
        if not self._load_recognition_components():
            self.status_label.config(text="Không thể bắt đầu nhận diện.", foreground="red")
            return # Không thể bắt đầu nếu model không tải được hoặc face detector lỗi

        self.check_type = check_type
        self.camera = initialize_camera() # Gọi hàm initialize_camera từ utils
        
        if self.camera is None: # Xử lý trường hợp camera không mở được
            messagebox.showerror("Lỗi Camera", "Không thể truy cập camera. Vui lòng kiểm tra kết nối hoặc xem log terminal.")
            self.status_label.config(text="Lỗi Camera", foreground="red")
            return

        self.is_recognizing = True
        self.status_label.config(text=f"Đang chờ {check_type}...", foreground="green")
        self.result_label.config(text="Đưa khuôn mặt vào camera", foreground="blue")

        self._recognize_face_loop()

    def _recognize_face_loop(self):
        """Vòng lặp để nhận diện khuôn mặt và cập nhật giao diện."""
        if not self.is_recognizing:
            self.video_label.config(image='') # Xóa hình ảnh khi dừng
            self.result_label.config(text="")
            return # Dừng vòng lặp nếu cờ is_recognizing là False
        
        if self.camera is None: # Đảm bảo self.camera không phải là None
            print("Lỗi: Camera đã bị đóng hoặc chưa được khởi tạo. Dừng vòng lặp nhận diện.")
            self.stop_recognition()
            return

        ret, img = self.camera.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ camera. Dừng nhận diện.")
            self.status_label.config(text="Lỗi đọc camera. Đã dừng.", foreground="red")
            self.stop_recognition()
            return

        img = cv2.flip(img, 1) # Flip theo chiều ngang
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detected_face_in_frame = False

        # Chỉ thực hiện phát hiện và nhận diện nếu faceCascade và recognizer đã được tải
        if self.faceCascade and self.recognizer:
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for (x, y, w, h) in faces:
                detected_face_in_frame = True
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Vùng ảnh khuôn mặt để dự đoán
                roi_gray = gray[y:y + h, x:x + w]
                # Đảm bảo roi_gray không rỗng và có kích thước hợp lệ
                if roi_gray.size > 0:
                    id_predicted, confidence = self.recognizer.predict(roi_gray)
                else:
                    id_predicted = -1 # ID không xác định
                    confidence = 100 # Độ tin cậy cao (không đáng tin cậy)

                original_user_id = self.id_mapping.get(id_predicted, "unknown")
                predicted_name = self.names.get(original_user_id, "Không xác định")

                current_time = time.time()
                last_recorded_time = self.last_recognized_time.get(original_user_id, 0)

                # Kiểm tra ngưỡng tin cậy VÀ cooldown
                if confidence < config.CONFIDENCE_THRESHOLD: # Nếu độ tự tin tốt (nhận diện thành công)
                    if (current_time - last_recorded_time) > self.recognition_cooldown_time:
                        # Đã nhận diện thành công và hết thời gian cooldown
                        if predicted_name == "Không xác định":
                            name_from_file = get_name_for_id(original_user_id)
                            if name_from_file:
                                self.names[original_user_id] = name_from_file
                                predicted_name = name_from_file
                            else:
                                predicted_name = "Không xác định"

                        if predicted_name != "Không xác định":
                            # **Đã chấm công thành công**
                            self.result_label.config(text=f"Đã {self.check_type}: {predicted_name}", foreground="blue")
                            self.status_label.config(text="Thành công!", foreground="green")
                            record_attendance(original_user_id, predicted_name, self.check_type)
                            messagebox.showinfo("Thành công", f"{predicted_name} đã {self.check_type} thành công!")
                            self.last_recognized_time[original_user_id] = current_time

                            # TỰ ĐỘNG TẮT CAMERA SAU KHI CHẤM CÔNG THÀNH CÔNG
                            self.stop_recognition()
                            return # Rất quan trọng để dừng vòng lặp sau khi gọi stop_recognition

                        else: # Nếu không thể lấy được tên
                            self.result_label.config(text="Không xác định", foreground="red")
                            self.status_label.config(text="Vui lòng thử lại", foreground="red")

                    else:
                        # Vẫn đang trong thời gian cooldown
                        self.result_label.config(text=f"Vui lòng chờ: {predicted_name}", foreground="orange")
                        self.status_label.config(text="Đang chờ cooldown", foreground="orange")
                        cv2.putText(img, predicted_name, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        cv2.putText(img, f"Conf: {confidence:.1f}%", (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)

                else: # Confidence quá cao hoặc không khớp
                    self.result_label.config(text="Không xác định", foreground="red")
                    self.status_label.config(text="Vui lòng thử lại", foreground="red")
                    predicted_name = "Không xác định"
                    cv2.putText(img, predicted_name, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.putText(img, f"Conf: {confidence:.1f}%", (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
        else: # Nếu faceCascade hoặc recognizer chưa được tải
            cv2.putText(img, "Lỗi: Khong the tai model hoac Face Detector!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.status_label.config(text="Lỗi: Chưa tải model/Face Detector.", foreground="red")
            self.result_label.config(text="Vui lòng kiểm tra file model và haarcascade.", foreground="red")


        if not detected_face_in_frame:
            # Nếu không có khuôn mặt nào được phát hiện trong khung hình
            self.result_label.config(text="Đưa khuôn mặt vào camera", foreground="blue")
            self.status_label.config(text="Đang chờ...", foreground="blue")

        # Hiển thị khung hình lên giao diện
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.video_label.img = img_tk
        self.video_label.config(image=img_tk)

        # Lặp lại sau 10ms
        self.video_label.after(10, self._recognize_face_loop)


    def stop_recognition(self):
        """Dừng quá trình nhận diện và giải phóng camera."""
        if self.is_recognizing:
            self.is_recognizing = False # Đặt cờ này trước khi giải phóng camera
            release_camera(self.camera)
            self.camera = None
            self.status_label.config(text="Sẵn sàng", foreground="blue")
            self.result_label.config(text="Đã dừng nhận diện")
            self.video_label.config(image='') # Xóa hình ảnh trên label
        else:
            print("Đã dừng nhận diện. Không cần thực hiện lại.")

    def perform_check_in(self):
        """Hàm xử lý khi nhấn nút Chấm Công (Check-in)."""
        self.start_recognition("Check-in")

    def perform_check_out(self):
        """Hàm xử lý khi nhấn nút Check-Out."""
        self.start_recognition("Check-Out")