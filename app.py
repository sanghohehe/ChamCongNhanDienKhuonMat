import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
import time

# Import từ các module đã tách
from admin.admin_functions import AdminFunctions
from main.main_functions import MainFunctions
from database.database_manager import record_attendance, load_attendance, save_attendance
from utils.face_recognizer_utils import get_name_for_id
import config # Import config.py

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống nhận dạng khuôn mặt")

        # Đảm bảo các thư mục cần thiết tồn tại
        os.makedirs(config.DATASET_PATH, exist_ok=True)
        os.makedirs(config.TRAINER_PATH, exist_ok=True)

        self.names = {} # Dictionary để lưu trữ user_id và tên người dùng
        self.load_existing_users()

        # 1. Tạo các widget GUI chung trước
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(main_frame)
        self.video_label.pack()

        self.result_label = ttk.Label(main_frame, text="", font=('Helvetica', 12))
        self.result_label.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(main_frame, text="Sẵn sàng", font=("Helvetica", 11), foreground="blue")
        self.status_label.pack(pady=5)

        # 2. Khởi tạo các module con SAU KHI các widget đã được tạo
        # Truyền các widget đã tạo vào các đối tượng AdminFunctions và MainFunctions
        self.admin_functions = AdminFunctions(
            self.root, self.video_label, self.status_label, self.result_label, self.names
        )
        self.main_functions = MainFunctions(
            self.root, self.video_label, self.status_label, self.result_label, self.names
        )

        # 3. Tạo các nút và các phần còn lại của giao diện người dùng
        self._create_buttons_ui(main_frame)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_existing_users(self):
        # Tải tên người dùng từ các file .name.txt trong thư mục dataset
        for filename in os.listdir(config.DATASET_PATH):
            if filename.endswith('.name.txt'):
                user_id = filename.split('.')[1]
                name = get_name_for_id(user_id)
                if name:
                    self.names[user_id] = name

    def _create_buttons_ui(self, parent_frame):
        """Tạo các nút và khung chứa nút cho giao diện chính."""
        # Nút hàng 1 (Chức năng Admin)
        admin_button_frame = ttk.Frame(parent_frame)
        admin_button_frame.pack(pady=10)

        btn_capture = tk.Button(admin_button_frame, text="📸 Chụp ảnh", command=self.admin_functions.start_capture_images,
                                bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
        btn_capture.pack(side=tk.LEFT, padx=15)

        btn_train = tk.Button(admin_button_frame, text="🧠 Train Model", command=self.admin_functions.train_recognition_model,
                              bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
        btn_train.pack(side=tk.LEFT, padx=15)

        btn_view_attendance = tk.Button(admin_button_frame, text="📊 Xem Chấm Công", command=self.admin_functions.view_attendance_records,
                                        bg="#9C27B0", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
        btn_view_attendance.pack(side=tk.LEFT, padx=15)

        btn_reports = tk.Button(admin_button_frame, text="📈 Thống kê - Báo cáo", command=self.admin_functions.show_report_window,
                                bg="#FF5722", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
        btn_reports.pack(side=tk.LEFT, padx=15)


        # Nút hàng 2 (Chức năng Main - Chấm công)
        main_button_frame = ttk.Frame(parent_frame)
        main_button_frame.pack(pady=10)

        btn_check_in = tk.Button(main_button_frame, text="🔍 Chấm Công (Check-in)", command=self.main_functions.perform_check_in,
                                 bg="#FF9800", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_check_in.pack(side=tk.LEFT, padx=15)

        btn_check_out = tk.Button(main_button_frame, text="🔓 Check-Out", command=self.main_functions.perform_check_out,
                                  bg="#E91E63", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_check_out.pack(side=tk.LEFT, padx=15)

    # Phương thức create_main_ui không cần thiết nữa vì đã tách logic
    # Bạn có thể xóa nó hoặc để trống nếu muốn
    # def create_main_ui(self):
    #     pass

    def stop_all_processes(self):
        """Dừng tất cả các tiến trình capture/recognition đang chạy."""
        if self.admin_functions.is_capturing:
            self.admin_functions.stop_capture()
        if self.main_functions.is_recognizing:
            self.main_functions.stop_recognition()

    def on_close(self):
        """Dừng camera và thoát ứng dụng khi đóng cửa sổ."""
        try:
            self.stop_all_processes()
            save_attendance(load_attendance()) # Đảm bảo lưu dữ liệu cuối cùng
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu lịch sử chấm công: {str(e)}")
        finally:
            self.root.quit()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()