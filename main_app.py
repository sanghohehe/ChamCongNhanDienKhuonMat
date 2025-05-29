import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime

# Import playsound cho âm thanh
from playsound import playsound

# Import các module đã tách
from database.database_manager import record_attendance, load_attendance, save_attendance
from utils.face_recognizer_utils import load_recognizer_model, load_id_mapping, get_name_for_id
from utils.camera_utils import load_face_detector, initialize_camera, release_camera
import config

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống chấm công bằng khuôn mặt")
        # Thay đổi kích thước cửa sổ để có chỗ cho bảng lịch sử
        self.root.geometry("600x550") 
        self.root.resizable(False, False)

        os.makedirs(config.DATASET_PATH, exist_ok=True)
        os.makedirs(config.TRAINER_PATH, exist_ok=True)
        os.makedirs(config.SOUNDS_FOLDER, exist_ok=True)

        self.names = {}
        self.id_mapping = {}
        self.face_detector = None
        self.recognizer = None
        self.load_existing_users()
        self.load_models()

        self.last_check_time = {} 
        # Dictionary để lưu thời gian check-in/out gần nhất của mỗi người
        # Key: original_user_id_str, Value: {"Name": "Tên", "CheckIn": "HH:MM:SS", "CheckOut": "HH:MM:SS"}
        self.latest_attendance = {}

        self.create_widgets()
        # Tải và hiển thị lịch sử chấm công ban đầu
        self._load_initial_latest_attendance() 

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_existing_users(self):
        """
        Tải tên người dùng từ file ánh xạ ID.
        self.id_mapping: numeric_id -> original_id_str
        self.names: numeric_id -> name (dùng để hiển thị)
        """
        self.id_mapping = load_id_mapping() 
        temp_names = {}
        for numeric_id, original_id_str in self.id_mapping.items():
            name = get_name_for_id(original_id_str) 
            if name and name != "Không xác định": 
                temp_names[numeric_id] = name
            else:
                temp_names[numeric_id] = original_id_str 
        self.names = temp_names
        print(f"DEBUG: Loaded Names mapping (numeric_id to name): {self.names}") 

    def load_models(self):
        try:
            self.face_detector = load_face_detector()
            self.recognizer = load_recognizer_model()
            print("Đã tải Face Detector và Recognizer model.")
        except Exception as e:
            messagebox.showerror("Lỗi tải model", f"Không thể tải model nhận diện hoặc bộ phát hiện khuôn mặt: {e}\n"
                                                 "Vui lòng đảm bảo bạn đã train model và file 'haarcascade_frontalface_default.xml' có tồn tại.")
            self.face_detector = None
            self.recognizer = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        title_label = ttk.Label(main_frame, text="HỆ THỐNG CHẤM CÔNG", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=20)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.checkin_button = ttk.Button(button_frame, text="CHẤM CÔNG", 
                                         command=lambda: self.open_camera_for_attendance("Check-in"),
                                         width=20, style='Big.TButton')
        self.checkin_button.pack(side=tk.LEFT, padx=15)

        self.checkout_button = ttk.Button(button_frame, text="CHECK-OUT", 
                                          command=lambda: self.open_camera_for_attendance("Check-out"),
                                          width=20, style='Big.TButton')
        self.checkout_button.pack(side=tk.LEFT, padx=15)
        
        style = ttk.Style()
        style.configure('Big.TButton', font=('Helvetica', 14, 'bold'), padding=10)

        self.status_label = ttk.Label(main_frame, text="Sẵn sàng để chấm công", font=("Helvetica", 11), foreground="blue")
        self.status_label.pack(pady=10)

        # --- Bảng lịch sử chấm công gần nhất ---
        history_frame = ttk.LabelFrame(main_frame, text="Lịch sử chấm công gần nhất", padding="10")
        history_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.attendance_table = ttk.Treeview(history_frame, columns=("Name", "CheckIn", "CheckOut"), show="headings", height=5) # height giới hạn số dòng hiển thị
        self.attendance_table.heading("Name", text="Tên")
        self.attendance_table.heading("CheckIn", text="Check-in gần nhất")
        self.attendance_table.heading("CheckOut", text="Check-out gần nhất")

        self.attendance_table.column("Name", width=120, anchor=tk.W)
        self.attendance_table.column("CheckIn", width=120, anchor=tk.CENTER)
        self.attendance_table.column("CheckOut", width=120, anchor=tk.CENTER)
        
        self.attendance_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar cho bảng
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.attendance_table.yview)
        self.attendance_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # --- Hết bảng lịch sử ---

    def play_sound(self, sound_file_path):
        """Phát file âm thanh."""
        if os.path.exists(sound_file_path):
            try:
                playsound(sound_file_path, False)
                print(f"Đã phát âm thanh: {sound_file_path}")
            except Exception as e:
                print(f"Lỗi khi phát âm thanh {sound_file_path}: {e}")
        else:
            print(f"File âm thanh không tồn tại: {sound_file_path}")

    def open_camera_for_attendance(self, check_type):
        if self.face_detector is None or self.recognizer is None:
            messagebox.showwarning("Lỗi", "Hệ thống chưa tải đủ model. Vui lòng kiểm tra lại cấu hình hoặc train model.")
            return

        self.checkin_button.config(state=tk.DISABLED)
        self.checkout_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Đang chờ chấm công {check_type}...", foreground="orange")
        
        camera_window = tk.Toplevel(self.root)
        camera_window.title(f"Camera Chấm công - {check_type}")
        camera_window.geometry("800x650")
        camera_window.resizable(False, False)
        
        camera_window.protocol("WM_DELETE_WINDOW", lambda: self.on_camera_close_attempt(camera_window))

        camera_frame = ttk.Frame(camera_window, padding="10")
        camera_frame.pack(expand=True, fill=tk.BOTH)

        video_label = ttk.Label(camera_frame, background="black")
        video_label.pack(side=tk.TOP, padx=5, pady=5)

        result_label = ttk.Label(camera_frame, text="Vui lòng đưa mặt vào khung hình", font=('Helvetica', 16, 'bold'))
        result_label.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.camera_is_running = True
        self.cam = initialize_camera()

        if self.cam is None:
            messagebox.showerror("Lỗi Camera", "Không thể truy cập camera. Vui lòng kiểm tra kết nối.")
            camera_window.destroy()
            self.checkin_button.config(state=tk.NORMAL)
            self.checkout_button.config(state=tk.NORMAL)
            self.status_label.config(text="Camera không khả dụng.", foreground="red")
            return

        self._update_camera_feed(camera_window, video_label, result_label, check_type)

    def on_camera_close_attempt(self, camera_window):
        if messagebox.askyesno("Xác nhận đóng", "Bạn có chắc chắn muốn thoát khỏi chế độ chấm công?"):
            self.stop_camera_feed(camera_window)
            self.status_label.config(text="Sẵn sàng để chấm công", foreground="blue")

    def _update_camera_feed(self, camera_window, video_label, result_label, check_type):
        if not self.camera_is_running or not self.cam.isOpened():
            return

        ret, img = self.cam.read()
        if not ret:
            result_label.config(text="Lỗi đọc khung hình từ camera.", foreground="red")
            camera_window.after(10, lambda: self._update_camera_feed(camera_window, video_label, result_label, check_type))
            return

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

        current_time = datetime.now()
        recognized_successfully = False

        for (x, y, w, h) in faces:
            color = (255, 0, 0)
            text = "Dang nhan dien..."

            try:
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size > 0:
                    recognized_id_numeric, confidence = self.recognizer.predict(roi_gray)
                else:
                    recognized_id_numeric = -1
                    confidence = 100

                # Dòng debug quan trọng: in ra ID số và độ tin cậy từ model
                # print(f"DEBUG: Recognized Numeric ID: {recognized_id_numeric}, Confidence: {confidence:.2f}")

                original_user_id_str = self.id_mapping.get(recognized_id_numeric, "unknown")
                # Dòng debug: in ra ID gốc (string) được ánh xạ từ ID số
                # print(f"DEBUG: Original User ID String from mapping: {original_user_id_str}")

                # ✅ DÒNG SỬA LỖI QUAN TRỌNG: Dùng recognized_id_numeric để lấy tên từ self.names
                predicted_name = self.names.get(recognized_id_numeric, "Không xác định") 
                # Dòng debug: in ra tên được dự đoán từ self.names
                # print(f"DEBUG: Predicted Name from self.names (using numeric ID): {predicted_name}")
                
                last_check_time_for_user = self.last_check_time.get(original_user_id_str)

                if confidence < config.CONFIDENCE_THRESHOLD:
                    if last_check_time_for_user is None or \
                       (current_time - last_check_time_for_user).total_seconds() > config.COOLDOWN_TIME:
                        
                        # ✅ THAY ĐỔI LOGIC PHÁT ÂM THANH Ở ĐÂY
                        if check_type == "Check-in":
                            self.play_sound(config.SUCCESS_SOUND) # Phát âm thanh Check-in
                        elif check_type == "Check-out":
                            self.play_sound(config.CHECKOUT_SOUND) # Phát âm thanh Check-out
                        
                        record_attendance(original_user_id_str, predicted_name, check_type)
                        self.last_check_time[original_user_id_str] = current_time
                        
                        result_label.config(text=f"{check_type} thành công: {predicted_name}", foreground="green")
                        color = (0, 255, 0)
                        text = f"{predicted_name} ({confidence:.0f}%)"
                        recognized_successfully = True

                        # Cập nhật latest_attendance và bảng
                        if original_user_id_str not in self.latest_attendance:
                            self.latest_attendance[original_user_id_str] = {"Name": predicted_name, "CheckIn": "-", "CheckOut": "-"}

                        if check_type == "Check-in":
                            self.latest_attendance[original_user_id_str]["CheckIn"] = current_time.strftime("%H:%M:%S")
                        elif check_type == "Check-out":
                            self.latest_attendance[original_user_id_str]["CheckOut"] = current_time.strftime("%H:%M:%S")
                        
                        self._update_attendance_table() # Cập nhật bảng
                        
                    else:
                        remaining_time = config.COOLDOWN_TIME - (current_time - last_check_time_for_user).total_seconds()
                        result_label.config(text=f"{predicted_name}: Cooldown {int(remaining_time)}s", foreground="orange")
                        color = (0, 255, 255)
                        text = f"{predicted_name} (Cooldown)"
                        
                else:
                    result_label.config(text=f"Khuôn mặt không xác định ({confidence:.0f}%)", foreground="red")
                    color = (0, 0, 255)
                    text = "Unknown"
                    if self.last_check_time.get('unknown', datetime.min) + pd.Timedelta(seconds=5) < current_time:
                        self.play_sound(config.UNKNOWN_SOUND)
                        self.last_check_time['unknown'] = current_time
                    
            except cv2.error as cv_err:
                print(f"Lỗi OpenCV khi nhận diện: {cv_err}")
                result_label.config(text="Lỗi nhận diện", foreground="red")
                color = (0, 0, 255)
                text = "Loi nhan dien"
            except Exception as e:
                print(f"Lỗi không xác định khi nhận diện: {e}")
                result_label.config(text="Lỗi hệ thống", foreground="red")
                color = (0, 0, 255)
                text = "Loi he thong"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        video_label.img_tk = img_tk
        video_label.config(image=img_tk)

        if recognized_successfully:
            messagebox.showinfo("Chấm công", result_label.cget("text") + "\n\nNhấn OK để quay lại.")
            self.stop_camera_feed(camera_window)
        else:
            camera_window.after(10, lambda: self._update_camera_feed(camera_window, video_label, result_label, check_type))

    def stop_camera_feed(self, camera_window):
        self.camera_is_running = False
        if self.cam:
            release_camera(self.cam)
            self.cam = None
            print("Đã giải phóng camera.")
        camera_window.destroy()
        self.checkin_button.config(state=tk.NORMAL)
        self.checkout_button.config(state=tk.NORMAL)
        self.status_label.config(text="Sẵn sàng để chấm công", foreground="blue")

    def _load_initial_latest_attendance(self):
        """Tải dữ liệu chấm công gần nhất từ file attendance.csv khi khởi động."""
        df = load_attendance() # Sử dụng hàm đã có
        if not df.empty:
            # Chuyển đổi các cột thời gian nếu chúng chưa phải datetime objects
            df['CheckInTime'] = pd.to_datetime(df['CheckInTime'], errors='coerce')
            df['CheckOutTime'] = pd.to_datetime(df['CheckOutTime'], errors='coerce')
            
            # Sắp xếp theo UserID và thời gian để lấy bản ghi gần nhất
            df_sorted = df.sort_values(by=['UserID', 'CheckInTime', 'CheckOutTime'], ascending=[True, False, False])

            for user_id_str in df_sorted['UserID'].unique():
                user_records = df_sorted[df_sorted['UserID'] == user_id_str]
                
                # Tìm check-in gần nhất
                latest_checkin_record = user_records.dropna(subset=['CheckInTime']).sort_values(by='CheckInTime', ascending=False).head(1)
                latest_checkin_time = latest_checkin_record['CheckInTime'].iloc[0].strftime("%H:%M:%S") if not latest_checkin_record.empty else "-"
                
                # Tìm check-out gần nhất
                latest_checkout_record = user_records.dropna(subset=['CheckOutTime']).sort_values(by='CheckOutTime', ascending=False).head(1)
                latest_checkout_time = latest_checkout_record['CheckOutTime'].iloc[0].strftime("%H:%M:%S") if not latest_checkout_record.empty else "-"

                # Lấy tên người dùng (sử dụng names mapping)
                # Chuyển user_id_str trở lại numeric_id để tìm trong self.names
                # Tìm key (numeric_id) trong id_mapping mà value là user_id_str
                numeric_id = next((k for k, v in self.id_mapping.items() if v == user_id_str), None)
                predicted_name = self.names.get(numeric_id, user_id_str) # Dùng user_id_str nếu không tìm thấy tên

                self.latest_attendance[user_id_str] = {
                    "Name": predicted_name,
                    "CheckIn": latest_checkin_time,
                    "CheckOut": latest_checkout_time
                }
        self._update_attendance_table() # Cập nhật bảng sau khi tải dữ liệu ban đầu

    def _update_attendance_table(self):
        """Cập nhật Treeview với dữ liệu chấm công gần nhất."""
        # Xóa tất cả các mục hiện có trong bảng
        for item in self.attendance_table.get_children():
            self.attendance_table.delete(item)

        # Thêm dữ liệu từ self.latest_attendance vào bảng
        # Sắp xếp theo tên cho dễ nhìn
        sorted_items = sorted(self.latest_attendance.items(), key=lambda item: item[1]['Name'])
        for user_id, data in sorted_items:
            self.attendance_table.insert("", tk.END, values=(data["Name"], data["CheckIn"], data["CheckOut"]))

    def on_close(self):
        print("Đang đóng ứng dụng chính...")
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy() 
        
        from database.database_manager import save_attendance 
        save_attendance() 
        self.root.quit()
        self.root.destroy()
        print("Ứng dụng đã đóng.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()