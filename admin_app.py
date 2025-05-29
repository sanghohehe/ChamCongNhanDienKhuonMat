# admin_app.py
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
import time

# Import từ các module đã tách
from admin.admin_functions import AdminFunctions
from database.database_manager import load_attendance, save_attendance
from utils.face_recognizer_utils import get_name_for_id
import config # Import config.py

class AdminApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống nhận dạng khuôn mặt - ADMIN PANEL")
        self.root.geometry("1200x800") # Kích thước cửa sổ gợi ý

        # Đảm bảo các thư mục cần thiết tồn tại
        os.makedirs(config.DATASET_PATH, exist_ok=True)
        os.makedirs(config.TRAINER_PATH, exist_ok=True)

        self.names = {} # Dictionary để lưu trữ user_id và tên người dùng
        self.load_existing_users()

        # Tạo các widget GUI chung cho AdminApp
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Khung chứa video và trạng thái
        video_status_frame = ttk.Frame(main_frame)
        video_status_frame.pack(side=tk.TOP, pady=10)

        self.video_label = ttk.Label(video_status_frame, text="Camera Feed", background="black") # Bỏ width và height
        self.video_label.pack(side=tk.TOP, padx=5, pady=5)

        self.result_label = ttk.Label(video_status_frame, text="", font=('Helvetica', 12))
        self.result_label.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.status_label = ttk.Label(video_status_frame, text="Sẵn sàng", font=("Helvetica", 11), foreground="blue")
        self.status_label.pack(side=tk.TOP, pady=5)


        # Khởi tạo AdminFunctions, truyền các widget đã tạo
        self.admin_functions = AdminFunctions(
            self.root, self.video_label, self.status_label, self.result_label, self.names
        )

        # Tạo các nút chức năng Admin
        self._create_admin_buttons_ui(main_frame)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_existing_users(self):
        # Tải tên người dùng từ các file .name.txt trong thư mục dataset
        for filename in os.listdir(config.DATASET_PATH):
            if filename.endswith('.name.txt'):
                user_id = filename.split('.')[1]
                name = get_name_for_id(user_id)
                if name:
                    self.names[user_id] = name

    def _create_admin_buttons_ui(self, parent_frame):
        """Tạo các nút chức năng dành cho Admin."""
        admin_button_frame = ttk.Frame(parent_frame)
        admin_button_frame.pack(side=tk.TOP, pady=20) # Đặt ở giữa và có khoảng cách

        btn_capture = tk.Button(admin_button_frame, text="📸 Chụp ảnh", command=self.admin_functions.start_capture_images,
                                bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_capture.grid(row=0, column=0, padx=10, pady=10)

        btn_train = tk.Button(admin_button_frame, text="🧠 Train Model", command=self.admin_functions.train_recognition_model,
                              bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_train.grid(row=0, column=1, padx=10, pady=10)

        btn_view_attendance = tk.Button(admin_button_frame, text="📊 Xem Chấm Công", command=self.admin_functions.view_attendance_records,
                                        bg="#9C27B0", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_view_attendance.grid(row=1, column=0, padx=10, pady=10)

        btn_reports = tk.Button(admin_button_frame, text="📈 Thống kê - Báo cáo", command=self.admin_functions.show_report_window,
                                bg="#FF5722", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_reports.grid(row=1, column=1, padx=10, pady=10)

        btn_manage_users = tk.Button(admin_button_frame, text="👥 Quản lý Người dùng", command=self.admin_functions.show_user_management_window,
                                     bg="#607D8B", fg="white", font=("Helvetica", 12, "bold"), width=25, height=2)
        btn_manage_users.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def on_close(self):
        """Dừng camera và thoát ứng dụng khi đóng cửa sổ."""
        try:
            if self.admin_functions.is_capturing:
                self.admin_functions.stop_capture()
            # Không cần stop main_functions ở đây vì AdminApp không chạy nó
            save_attendance(load_attendance()) # Đảm bảo lưu dữ liệu cuối cùng
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu lịch sử chấm công: {str(e)}")
        finally:
            self.root.quit()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdminApp(root)
    root.mainloop()