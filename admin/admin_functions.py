# admin/admin_functions.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import shutil
import pandas as pd
import datetime
from PIL import Image, ImageTk # <<< THÊM DÒNG NÀY ĐỂ XỬ LÝ ẢNH

import config
from utils.face_recognizer_utils import (
    load_face_detector, collect_dataset, train_recognizer,
    load_recognizer_model, load_id_mapping, save_id_mapping, get_name_for_id, save_name_for_id
)
from database.database_manager import (
    load_attendance, save_attendance, summarize_checkin_checkout,
    update_user_name_in_attendance # <<< ĐẢM BẢO DÒNG NÀY CÓ Ở ĐÂY
)

import matplotlib.pyplot as plt # <<< THÊM DÒNG NÀY
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # <<< THÊM DÒNG NÀY

from report_window import ReportWindow 

class AdminFunctions:
    def __init__(self, master_root, video_label, status_label, result_label, names_dict):
        self.master_root = master_root
        self.video_label = video_label
        self.status_label = status_label
        self.result_label = result_label
        self.names = names_dict 

        self.cap = None 
        self.is_capturing = False
        self.face_detector = load_face_detector()
        self.recognizer = load_recognizer_model()
        self.id_mapping = load_id_mapping()

        self._load_names_from_id_mapping() 

    def _load_names_from_id_mapping(self):
        """Tải tên người dùng từ id_mapping.txt vào self.names."""
        for numeric_id, original_id_name in self.id_mapping.items():
            parts = original_id_name.split('_', 1)
            user_id_str = parts[0]
            name = parts[1] if len(parts) > 1 else "Unknown" 
            self.names[user_id_str] = name 

    def start_capture_images(self):
        self.stop_capture() 

        dialog = tk.Toplevel(self.master_root)
        dialog.title("Thu thập ảnh người dùng mới")
        dialog.geometry("400x200")
        dialog.transient(self.master_root)
        dialog.grab_set()

        ttk.Label(dialog, text="Mã định danh (UserID, VD: NV001):").pack(pady=5)
        user_id_entry = ttk.Entry(dialog, width=30)
        user_id_entry.pack(pady=5)

        ttk.Label(dialog, text="Tên người dùng:").pack(pady=5)
        user_name_entry = ttk.Entry(dialog, width=30)
        user_name_entry.pack(pady=5)

        def initiate_capture():
            user_id = user_id_entry.get().strip()
            user_name = user_name_entry.get().strip()
            if not user_id or not user_name:
                messagebox.showwarning("Cảnh báo", "Vui lòng nhập Mã định danh và Tên người dùng.")
                return

            if self.face_detector is None:
                messagebox.showerror("Lỗi", "Bộ phát hiện khuôn mặt chưa được tải.")
                dialog.destroy()
                return
            
            for num_id, original_id_name in self.id_mapping.items():
                if original_id_name.startswith(user_id + '_'): 
                    messagebox.showwarning("Trùng lặp", f"Mã định danh '{user_id}' đã tồn tại. Vui lòng chọn mã khác.")
                    return

            dialog.destroy() 

            messagebox.showinfo("Thu thập dữ liệu", f"Bắt đầu thu thập ảnh cho {user_name} (ID: {user_id}).\n"
                                                    "Vui lòng đưa khuôn mặt vào camera. Nhấn 'q' để dừng.")
            try:
                sample_count = collect_dataset(user_id, user_name, self.face_detector)
                if sample_count > 0:
                    save_name_for_id(user_id, user_name)
                    messagebox.showinfo("Hoàn thành", f"Đã thu thập {sample_count} ảnh cho {user_name}.\n"
                                                      "Vui lòng nhấn 'Train Model' để huấn luyện lại hệ thống.")
                else:
                    messagebox.showwarning("Cảnh báo", "Không thu thập được ảnh nào. Vui lòng thử lại.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi trong quá trình thu thập ảnh: {e}")
            
            self._update_main_window_labels("Sẵn sàng") 

        ttk.Button(dialog, text="Bắt đầu", command=initiate_capture).pack(pady=10)
        dialog.focus_set()

    def train_recognition_model(self):
        self.stop_capture() 

        messagebox.showinfo("Huấn luyện Model", "Bắt đầu huấn luyện model. Quá trình này có thể mất một lúc...")
        self._update_main_window_labels("Đang huấn luyện model...")
        
        try:
            self.id_mapping = load_id_mapping() 
            train_recognizer(self.face_detector)
            
            self.recognizer = load_recognizer_model()
            self.id_mapping = load_id_mapping()
            self._load_names_from_id_mapping() 

            messagebox.showinfo("Thành công", "Model nhận diện đã được huấn luyện lại thành công!")
            self._update_main_window_labels("Model đã sẵn sàng.")
        except ValueError as ve:
            messagebox.showwarning("Cảnh báo", f"Lỗi huấn luyện: {ve}\n"
                                             "Vui lòng đảm bảo có ít nhất một người dùng với đủ ảnh để huấn luyện.")
            self._update_main_window_labels("Lỗi huấn luyện.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong quá trình huấn luyện model: {e}")
            self._update_main_window_labels("Lỗi huấn luyện.")

    def view_attendance_records(self):
        """Hiển thị cửa sổ để xem và quản lý dữ liệu chấm công."""
        self.stop_capture() 

        attendance_window = tk.Toplevel(self.master_root)
        attendance_window.title("Quản lý Chấm công")
        attendance_window.geometry("1150x600") 
        attendance_window.transient(self.master_root)
        attendance_window.grab_set()

        tree_frame = ttk.Frame(attendance_window)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("Timestamp", "UserID", "Name", "CheckInTime", "CheckOutTime", "Status")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        tree.heading("Timestamp", text="Thời gian")
        tree.heading("UserID", text="Mã NV")
        tree.heading("Name", text="Tên")
        tree.heading("CheckInTime", text="Check-in")
        tree.heading("CheckOutTime", text="Check-out")
        tree.heading("Status", text="Trạng thái") 

        tree.column("Timestamp", width=150, anchor=tk.CENTER)
        tree.column("UserID", width=80, anchor=tk.CENTER)
        tree.column("Name", width=120, anchor=tk.W)
        tree.column("CheckInTime", width=120, anchor=tk.CENTER)
        tree.column("CheckOutTime", width=120, anchor=tk.CENTER)
        tree.column("Status", width=150, anchor=tk.CENTER) 
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

        def load_data():
            for i in tree.get_children():
                tree.delete(i)
            df = load_attendance()
            if not df.empty:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                if 'CheckInTime' in df.columns:
                    df['CheckInTime'] = pd.to_datetime(df['CheckInTime'], errors='coerce')
                if 'CheckOutTime' in df.columns:
                    df['CheckOutTime'] = pd.to_datetime(df['CheckOutTime'], errors='coerce')

                df = df.sort_values(by=['UserID', 'Timestamp']).reset_index(drop=True)

                display_data = []
                for index, row in df.iterrows():
                    current_status = ""
                    if row['CheckType'] == "Check-in":
                        if pd.notna(row['CheckOutTime']):
                            current_status = "Đã hoàn thành"
                        else:
                            current_status = "Đang làm việc"
                    elif row['CheckType'] == "Check-out":
                        if pd.notna(row['CheckInTime']): 
                            current_status = "Đã hoàn thành"
                        else: 
                            current_status = "Chỉ Check-out (Không Check-in)"

                    display_data.append((
                        row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        row['UserID'],
                        row['Name'],
                        row['CheckInTime'].strftime('%H:%M:%S') if pd.notna(row['CheckInTime']) else '',
                        row['CheckOutTime'].strftime('%H:%M:%S') if pd.notna(row['CheckOutTime']) else '',
                        current_status
                    ))
                
                for row_data in display_data:
                    tree.insert("", tk.END, values=row_data)
        
        load_data()

        button_panel_frame = ttk.Frame(attendance_window, width=150) 
        button_panel_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(button_panel_frame, text="Làm mới", command=load_data).pack(pady=5, fill=tk.X) 
        ttk.Button(button_panel_frame, text="Xóa bản ghi đã chọn", command=lambda: self._delete_selected_attendance_record(tree, load_data)).pack(pady=5, fill=tk.X)
        ttk.Button(button_panel_frame, text="Xóa tất cả", command=lambda: self._clear_all_attendance_records(load_data)).pack(pady=5, fill=tk.X)
        ttk.Button(button_panel_frame, text="Đóng", command=attendance_window.destroy).pack(side=tk.BOTTOM, pady=5, fill=tk.X)


    # Chuyển các hàm xóa bản ghi chấm công thành phương thức của lớp
    def _delete_selected_attendance_record(self, tree_view, refresh_func):
        selected_item = tree_view.selection()
        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một bản ghi để xóa.")
            return

        if messagebox.askyesno("Xác nhận xóa", "Bạn có chắc chắn muốn xóa bản ghi này?"):
            values = tree_view.item(selected_item, 'values')
            timestamp_str = values[0]
            user_id_to_delete = values[1]

            df = load_attendance()
            timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            df_filtered = df[(df['UserID'] == user_id_to_delete) & (df['Timestamp'] == timestamp_dt)]
            
            if not df_filtered.empty:
                df = df.drop(df_filtered.index)
                save_attendance(df)
                messagebox.showinfo("Thành công", "Đã xóa bản ghi.")
                refresh_func()
            else:
                messagebox.showwarning("Lỗi", "Không tìm thấy bản ghi để xóa.")

    def _clear_all_attendance_records(self, refresh_func):
        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa TẤT CẢ các bản ghi chấm công?"):
            df = pd.DataFrame(columns=['UserID', 'Name', 'Timestamp', 'CheckType', 'CheckInTime', 'CheckOutTime'])
            save_attendance(df)
            messagebox.showinfo("Thành công", "Đã xóa tất cả các bản ghi chấm công.")
            refresh_func()

    def show_report_window(self):
        self.stop_capture()
        ReportWindow(self.master_root) 

    # --- HÀM QUẢN LÝ NGƯỜI DÙNG MỚI/CẬP NHẬT ---
    def show_user_management_window(self):
        """Hiển thị cửa sổ quản lý người dùng (thêm sửa xóa, xem ảnh)."""
        self.stop_capture()

        user_management_window = tk.Toplevel(self.master_root)
        user_management_window.title("Quản lý người dùng")
        user_management_window.geometry("750x450") # Tăng kích thước cửa sổ
        user_management_window.transient(self.master_root)
        user_management_window.grab_set()

        # Frame cho Treeview
        tree_frame = ttk.Frame(user_management_window)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("UserID", "Name")
        
        # <<< ĐÂY LÀ DÒNG BẠN CẦN THAY ĐỔI ĐỂ GÁN CHO self.user_tree >>>
        self.user_tree = ttk.Treeview(tree_frame, columns=columns, show="headings") 
        
        # <<< TẤT CẢ CÁC DÒNG SỬ DỤNG 'tree' TRƯỚC ĐÓ BÂY GIỜ PHẢI DÙNG 'self.user_tree' >>>
        self.user_tree.heading("UserID", text="Mã NV")
        self.user_tree.heading("Name", text="Tên")
        self.user_tree.column("UserID", width=150, anchor=tk.CENTER)
        self.user_tree.column("Name", width=250, anchor=tk.W)
        self.user_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.user_tree.bind('<<TreeviewSelect>>', self._on_user_tree_select)

        # Thanh cuộn cũng phải liên kết với self.user_tree
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.user_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.user_tree.configure(yscrollcommand=scrollbar.set)

        def load_users():
            # Thay đổi 'tree' thành 'self.user_tree' trong hàm lồng này
            for i in self.user_tree.get_children():
                self.user_tree.delete(i)
            self.id_mapping = load_id_mapping()
            self._load_names_from_id_mapping() 
            
            sorted_user_ids = sorted(self.names.keys())
            for user_id in sorted_user_ids:
                name = self.names[user_id]
                self.user_tree.insert("", tk.END, values=(user_id, name))
        
        load_users()
       

        # Frame cho các nút chức năng (bên phải Treeview)
        button_panel_frame = ttk.Frame(user_management_window, width=180) # Đặt chiều rộng cố định
        button_panel_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Nút Sửa
        ttk.Button(button_panel_frame, text="Sửa thông tin người dùng", 
                   command=lambda: self._edit_user_info_dialog(self.user_tree, load_users)).pack(pady=5, fill=tk.X)
        
        # Nút Xóa
        ttk.Button(button_panel_frame, text="Xóa người dùng đã chọn", 
                   command=lambda: self._delete_user_from_system(self.user_tree, load_users)).pack(pady=5, fill=tk.X)
        
        # Nút Xem ảnh
        ttk.Button(button_panel_frame, text="Xem ảnh người dùng", 
                   command=lambda: self._view_user_images_dialog(self.user_tree)).pack(pady=5, fill=tk.X)
        
        btn_view_user_report = ttk.Button(button_panel_frame, text="Báo cáo người dùng", command=self._view_user_report)
        btn_view_user_report.pack(pady=5, fill=tk.X) # Dùng fill=tk.X để đồng bộ với các nút khác


        # Nút Đóng
        ttk.Button(button_panel_frame, text="Đóng", command=user_management_window.destroy).pack(side=tk.BOTTOM, pady=5, fill=tk.X)

    def _edit_user_info_dialog(self, tree_view, refresh_func):
        selected_item = tree_view.selection()
        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một người dùng để sửa.")
            return

        user_id_to_edit = tree_view.item(selected_item, 'values')[0]
        current_name = tree_view.item(selected_item, 'values')[1]

        edit_dialog = tk.Toplevel(self.master_root)
        edit_dialog.title(f"Sửa thông tin người dùng: {user_id_to_edit}")
        edit_dialog.geometry("350x180")
        edit_dialog.transient(self.master_root)
        edit_dialog.grab_set()

        ttk.Label(edit_dialog, text="Mã định danh (UserID):").pack(pady=5)
        user_id_label = ttk.Label(edit_dialog, text=user_id_to_edit, font=('Arial', 10, 'bold'))
        user_id_label.pack(pady=2)

        ttk.Label(edit_dialog, text="Tên người dùng mới:").pack(pady=5)
        new_name_entry = ttk.Entry(edit_dialog, width=30)
        new_name_entry.insert(0, current_name)
        new_name_entry.pack(pady=5)

        def save_changes():
            new_name = new_name_entry.get().strip()
            if not new_name:
                messagebox.showwarning("Cảnh báo", "Tên người dùng không được để trống.")
                return
            
            if new_name == current_name:
                messagebox.showinfo("Thông báo", "Không có thay đổi nào được thực hiện.")
                edit_dialog.destroy()
                return

            try:
                # 1. Tìm numeric_id và old_original_id_name
                numeric_id = None
                old_original_id_name = None # e.g., NV001_NguyenVanA
                current_id_mapping = load_id_mapping() # Load lại để đảm bảo mới nhất
                
                for num_id, original_id_name_val in current_id_mapping.items():
                    if original_id_name_val.startswith(user_id_to_edit + '_'):
                        numeric_id = num_id
                        old_original_id_name = original_id_name_val
                        break
                
                if numeric_id is None:
                    messagebox.showerror("Lỗi", "Không tìm thấy người dùng trong ID mapping.")
                    edit_dialog.destroy()
                    return

                new_original_id_name = f"{user_id_to_edit}_{new_name}"

                # 2. Cập nhật tên thư mục trong dataset (nếu tên trong thư mục thay đổi)
                old_folder_path = os.path.join(config.DATASET_PATH, old_original_id_name)
                new_folder_path = os.path.join(config.DATASET_PATH, new_original_id_name)

                if os.path.exists(old_folder_path) and old_folder_path != new_folder_path:
                    shutil.move(old_folder_path, new_folder_path)
                    print(f"Đã đổi tên thư mục dataset từ '{old_folder_path}' thành '{new_folder_path}'")
                elif not os.path.exists(old_folder_path):
                    print(f"Cảnh báo: Thư mục dataset cũ '{old_folder_path}' không tồn tại. Không đổi tên.")

                # 3. Cập nhật file .name.txt
                save_name_for_id(user_id_to_edit, new_name)

                # 4. Cập nhật id_mapping.txt
                current_id_mapping[numeric_id] = new_original_id_name
                save_id_mapping(current_id_mapping)
                self._load_names_from_id_mapping() # Cập nhật lại self.names

                # 5. Cập nhật các bản ghi chấm công
                update_user_name_in_attendance(user_id_to_edit, new_name)

                messagebox.showinfo("Thành công", f"Đã cập nhật tên người dùng cho ID '{user_id_to_edit}' thành '{new_name}'.\n"
                                                  "Vui lòng huấn luyện lại model để thay đổi có hiệu lực trong nhận diện.")
                edit_dialog.destroy()
                refresh_func() # Làm mới Treeview chính
                self._update_main_window_labels("Thông tin người dùng đã cập nhật. Cần Train lại Model.")

            except Exception as e:
                messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi cập nhật: {e}")

        ttk.Button(edit_dialog, text="Lưu thay đổi", command=save_changes).pack(pady=10)
        edit_dialog.focus_set()

    def _delete_user_from_system(self, tree_view, refresh_func):
        selected_item = tree_view.selection()
        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một người dùng để xóa.")
            return

        user_id_to_delete = tree_view.item(selected_item, 'values')[0]
        user_name_to_delete = tree_view.item(selected_item, 'values')[1]

        if not messagebox.askyesno("Xác nhận xóa", f"Bạn có chắc chắn muốn xóa '{user_name_to_delete}' (ID: {user_id_to_delete}) khỏi hệ thống?\n"
                                                      "Hành động này sẽ xóa:\n"
                                                      "- Ảnh khuôn mặt trong dataset\n"
                                                      "- Bản ghi tên (.name.txt)\n"
                                                      "- Các bản ghi chấm công liên quan\n"
                                                      "Bạn sẽ cần huấn luyện lại model sau khi xóa."):
                return

        try:
            # Bước 1: Xóa thư mục ảnh và file .name.txt trong dataset
            folder_name_to_delete = None
            for num_id, original_id_name in load_id_mapping().items(): 
                if original_id_name.startswith(user_id_to_delete + '_'):
                    folder_name_to_delete = original_id_name
                    break
            
            if folder_name_to_delete:
                dataset_path_to_delete = os.path.join(config.DATASET_PATH, folder_name_to_delete)
                if os.path.exists(dataset_path_to_delete):
                    shutil.rmtree(dataset_path_to_delete)
                    print(f"Đã xóa thư mục dataset: {dataset_path_to_delete}")
            else:
                print(f"Không tìm thấy thư mục dataset cho ID: {user_id_to_delete}")
            
            name_file_path = os.path.join(config.DATASET_PATH, f"User.{user_id_to_delete}.name.txt")
            if os.path.exists(name_file_path):
                os.remove(name_file_path)
                print(f"Đã xóa file tên: {name_file_path}")

            # Bước 2: Xóa người dùng khỏi id_mapping.txt (tải lại, xóa, lưu lại)
            current_id_mapping = load_id_mapping()
            numeric_id_found = None
            for num_id, original_id_name in list(current_id_mapping.items()):
                if original_id_name.startswith(user_id_to_delete + '_'):
                    numeric_id_found = num_id
                    del current_id_mapping[num_id]
                    break
            if numeric_id_found is not None:
                save_id_mapping(current_id_mapping)
                print(f"Đã xóa ID {user_id_to_delete} khỏi id_mapping.")
            else:
                print(f"Không tìm thấy {user_id_to_delete} trong id_mapping để xóa.")

            # Bước 3: Xóa các bản ghi chấm công của người dùng này
            df_attendance = load_attendance()
            initial_rows = len(df_attendance)
            df_attendance = df_attendance[df_attendance['UserID'] != user_id_to_delete]
            if len(df_attendance) < initial_rows:
                save_attendance(df_attendance)
                print(f"Đã xóa các bản ghi chấm công của {user_id_to_delete}.")

            messagebox.showinfo("Thành công", f"Đã xóa '{user_name_to_delete}' (ID: {user_id_to_delete}) thành công.\n"
                                              "Vui lòng huấn luyện lại model để thay đổi có hiệu lực.")
            refresh_func() 
            self._update_main_window_labels("Người dùng đã xóa. Cần Train lại Model.")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi xóa người dùng: {e}")

    def _view_user_images_dialog(self, tree_view):
        selected_item = tree_view.selection()
        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một người dùng để xem ảnh.")
            return

        user_id = tree_view.item(selected_item, 'values')[0]
        user_name = tree_view.item(selected_item, 'values')[1]

        print(f"DEBUG: Selected UserID: {user_id}, Name: {user_name}")

        # Tìm tên thư mục đầy đủ (UserID_Name) từ id_mapping
        user_folder_name = None
        # Tải lại id_mapping để đảm bảo dữ liệu mới nhất
        current_id_mapping = load_id_mapping() 
        print(f"DEBUG: Current ID Mapping loaded: {current_id_mapping}")

        # Duyệt qua id_mapping để tìm folder_name dựa trên user_id
        for numeric_id, original_id_name in current_id_mapping.items():
            if original_id_name.startswith(user_id + '_'):
                user_folder_name = original_id_name
                break
        
        print(f"DEBUG: Determined user_folder_name from id_mapping: {user_folder_name}")

        if not user_folder_name:
            # Fallback: nếu không tìm thấy trong id_mapping (có thể do lỗi dữ liệu hoặc chưa train lại)
            # Thử tạo tên thư mục theo định dạng phổ biến UserID_Name
            user_folder_name_fallback = f"{user_id}_{user_name}"
            potential_path_fallback = os.path.join(config.DATASET_PATH, user_folder_name_fallback)
            if os.path.exists(potential_path_fallback):
                user_folder_name = user_folder_name_fallback
                print(f"DEBUG: Using fallback user_folder_name: {user_folder_name}")
            else:
                messagebox.showerror("Lỗi", f"Không tìm thấy thư mục ảnh cho người dùng ID: {user_id}. Vui lòng đảm bảo người dùng có ảnh và model đã được huấn luyện lại.")
                return

        user_image_path = os.path.join(config.DATASET_PATH, user_folder_name)
        print(f"DEBUG: Full user image path: {user_image_path}")

        if not os.path.exists(user_image_path):
            messagebox.showinfo("Thông báo", f"Thư mục ảnh không tồn tại cho người dùng '{user_name}' (ID: {user_id}): {user_image_path}")
            return
        
        image_files = sorted([f for f in os.listdir(user_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"DEBUG: Found {len(image_files)} image files in {user_image_path}")

        if not image_files:
            messagebox.showinfo("Thông báo", f"Không có ảnh nào cho người dùng '{user_name}' (ID: {user_id}) trong thư mục: {user_image_path}")
            return

        image_viewer_dialog = tk.Toplevel(self.master_root)
        image_viewer_dialog.title(f"Ảnh của: {user_name} (ID: {user_id})")
        image_viewer_dialog.geometry("800x600")
        image_viewer_dialog.transient(self.master_root)
        image_viewer_dialog.grab_set()

        canvas = tk.Canvas(image_viewer_dialog, bg="white")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(image_viewer_dialog, orient="vertical", command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill="y")
        canvas.configure(yscrollcommand=v_scrollbar.set)
        
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion = canvas.bbox("all")))

        frame_in_canvas = ttk.Frame(canvas)
        canvas.create_window((0,0), window=frame_in_canvas, anchor="nw")

        row_count = 0
        col_count = 0
        max_cols = 4 # Hiển thị tối đa 4 ảnh mỗi hàng
        photo_references = [] # Danh sách để giữ tham chiếu đến PhotoImage objects

        for img_file in image_files:
            img_path = os.path.join(user_image_path, img_file)
            print(f"DEBUG: Attempting to load image: {img_path}")
            try:
                img = Image.open(img_path)
                original_width, original_height = img.size
                aspect_ratio = original_width / original_height
                
                fixed_height = 150
                fixed_width = int(fixed_height * aspect_ratio)

                if fixed_width > 200:
                    fixed_width = 200
                    fixed_height = int(fixed_width / aspect_ratio)

                img = img.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                photo_references.append(photo) # Dòng này đã có và cần được giữ

                label = ttk.Label(frame_in_canvas, image=photo)
                # >>> THÊM DÒNG NÀY VÀO NGAY DƯỚI DÒNG TRÊN <<<
                label.image = photo # Giữ một tham chiếu trực tiếp trên label
                label.grid(row=row_count, column=col_count, padx=5, pady=5)
                
                filename_label = ttk.Label(frame_in_canvas, text=img_file, font=('Arial', 7))
                filename_label.grid(row=row_count + 1, column=col_count, padx=5, pady=2)


                col_count += 1
                if col_count >= max_cols:
                    col_count = 0
                    row_count += 2
            except Exception as e:
                print(f"Lỗi khi tải hoặc hiển thị ảnh {img_file} từ {img_path}: {e}")

        # Nút đóng cho cửa sổ xem ảnh
        ttk.Button(image_viewer_dialog, text="Đóng", command=image_viewer_dialog.destroy).pack(pady=10)

        # Cập nhật scroll region sau khi tất cả ảnh được tải
        frame_in_canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def stop_capture(self):
        if self.is_capturing:
            self.is_capturing = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')
            self.video_label.image = None
            self._update_main_window_labels("Camera đã tắt.")

    def _update_main_window_labels(self, status_text="", result_text=""):
        self.status_label.config(text=status_text)
        self.result_label.config(text=result_text)

    def _view_user_report(self):
        selected_item = self.user_tree.selection()
        # print(f"DEBUG: selected_item = {selected_item}") # Có thể bỏ comment để debug

        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một người dùng để xem báo cáo.")
            return

        values = self.user_tree.item(selected_item[0], 'values')
        user_id_from_tree = values[0].strip() # Lấy ID từ Treeview và loại bỏ khoảng trắng
        user_name = values[1]

        # print(f"DEBUG: User selected from Treeview: ID='{user_id_from_tree}', Name='{user_name}'") # Có thể bỏ comment để debug

        df_attendance = load_attendance() # Tải toàn bộ dữ liệu chấm công

        if not df_attendance.empty:
            df_attendance['UserID'] = df_attendance['UserID'].astype(str) # Đảm bảo cột 'UserID' là kiểu chuỗi
            # print(f"DEBUG: Loaded attendance data. Columns: {df_attendance.columns}") # Có thể bỏ comment để debug
            # print(f"DEBUG: Unique UserIDs in loaded attendance data (original): {df_attendance['UserID'].unique()}") # Có thể bỏ comment để debug

            # Lấy UserID từ Treeview. UserID trong Treeview có dạng "ID_Name" (e.g., SS2_HoSang)
            # Chúng ta cần lấy phần ID số để so sánh với UserID_Extracted trong df_attendance.
            user_id_numeric_for_comparison = user_id_from_tree.split('_')[0].strip().upper()
            # print(f"DEBUG: User ID from Treeview (numeric for comparison): '{user_id_numeric_for_comparison}'") # Có thể bỏ comment để debug

            # Tạo một cột tạm thời 'ExtractedUserID' bằng cách cắt chuỗi 'UserID' tại dấu '_'
            # và chuyển nó thành chữ hoa để so sánh.
            df_attendance['ExtractedUserID'] = df_attendance['UserID'].apply(lambda x: x.split('_')[0]).str.upper()
            # print(f"DEBUG: UserIDs extracted from attendance data (normalized): {df_attendance['ExtractedUserID'].unique()}") # Có thể bỏ comment để debug

            # Lọc DataFrame dựa trên cột 'ExtractedUserID' mới tạo này
            df_user_detailed_attendance = df_attendance[df_attendance['ExtractedUserID'] == user_id_numeric_for_comparison]
        else:
            df_user_detailed_attendance = pd.DataFrame() 

        # print(f"DEBUG: Filtered attendance for UserID '{user_id_from_tree}'. Rows found: {df_user_detailed_attendance.shape[0]}") # Có thể bỏ comment để debug

        if df_user_detailed_attendance.empty:
            messagebox.showinfo("Thông báo", f"Không có dữ liệu chấm công chi tiết nào cho người dùng '{user_name}' (ID: {user_id_from_tree}).")
            return

        # Sử dụng hàm summarize_checkin_checkout để tạo báo cáo tổng hợp cho người dùng này
        df_user_summary = summarize_checkin_checkout(df_user_detailed_attendance.copy()) # Truyền bản sao để tránh cảnh báo

        if df_user_summary.empty:
            messagebox.showinfo("Thông báo", f"Không có dữ liệu báo cáo tổng hợp cho người dùng '{user_name}' (ID: {user_id_from_tree}).")
            return

        # Hiển thị báo cáo trong một cửa sổ mới
        # TRUYỀN THÊM df_user_detailed_attendance (dữ liệu chi tiết đã lọc)
        self._display_single_user_report_window(user_name, df_user_summary, df_user_detailed_attendance)

    def _display_single_user_report_window(self, user_name, df_summary, df_user_detailed_attendance):
        report_window = tk.Toplevel(self.master_root)
        report_window.title(f"Báo cáo Chấm công của {user_name}")
        report_window.geometry("1400x700") # Kích thước lớn hơn để chứa biểu đồ
        report_window.transient(self.master_root)
        report_window.grab_set()

        # Tạo một khung chính để chia thành 2 phần: Bảng và Biểu đồ
        main_frame = ttk.Frame(report_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Phần bên trái: Bảng báo cáo tổng hợp
        report_table_frame = ttk.Frame(main_frame)
        report_table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(report_table_frame, text=f"Báo cáo Tổng hợp cho: {user_name}", font=("Arial", 14, "bold")).pack(pady=5)

        columns = df_summary.columns.tolist()
        tree = ttk.Treeview(report_table_frame, columns=columns, show="headings")

        # Định nghĩa tiêu đề cột và độ rộng
        headings_map = {
            'UserID': 'Mã NV',
            'Name': 'Tên',
            'TotalCheckIn': 'Tổng Check-in',
            'TotalCheckOut': 'Tổng Check-out',
            'AvgWorkDuration': 'TG làm việc TB',
            'Status': 'Trạng thái'
        }
        for col in columns:
            tree.heading(col, text=headings_map.get(col, col))
            if col == 'UserID': tree.column(col, width=100, anchor=tk.CENTER)
            elif col == 'Name': tree.column(col, width=120, anchor=tk.W)
            elif col == 'TotalCheckIn' or col == 'TotalCheckOut': tree.column(col, width=90, anchor=tk.CENTER)
            elif col == 'AvgWorkDuration': tree.column(col, width=120, anchor=tk.CENTER)
            elif col == 'Status': tree.column(col, width=180, anchor=tk.W)
            else: tree.column(col, width=80, anchor=tk.W)

        # Đổ dữ liệu vào Treeview
        for index, row in df_summary.iterrows():
            tree.insert("", tk.END, values=row.tolist())
        
        tree.pack(fill=tk.BOTH, expand=True)

        # Nút đóng cho cửa sổ báo cáo này
        ttk.Button(report_table_frame, text="Đóng", command=report_window.destroy).pack(pady=10)


        # --- PHẦN BỔ SUNG: BIỂU ĐỒ ---
        chart_container_frame = ttk.Frame(main_frame)
        chart_container_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(chart_container_frame, text="Biểu đồ Chấm công", font=("Arial", 14, "bold")).pack(pady=5)

        # Sử dụng df_user_detailed_attendance đã được truyền vào
        # print(f"DEBUG: df_user_detailed_attendance in display func head:\n{df_user_detailed_attendance.head()}") # Có thể bỏ comment để debug
        # print(f"DEBUG: df_user_detailed_attendance in display func shape: {df_user_detailed_attendance.shape}") # Có thể bỏ comment để debug

        # Tính toán các chỉ số cần thiết cho biểu đồ
        metrics = self._calculate_attendance_metrics(df_user_detailed_attendance)

        # --- BIỂU ĐỒ 1: TỔNG GIỜ LÀM VIỆC MỖI NGÀY ---
        if not metrics['daily_work_hours'].empty:
            fig1, ax1 = plt.subplots(figsize=(6, 3)) # Kích thước biểu đồ
            metrics['daily_work_hours'].plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Tổng giờ làm việc mỗi ngày', fontsize=10)
            ax1.set_xlabel('Ngày', fontsize=8)
            ax1.set_ylabel('Tổng giờ (giờ)', fontsize=8)
            ax1.tick_params(axis='x', rotation=45, labelsize=7)
            ax1.tick_params(axis='y', labelsize=7)
            plt.tight_layout() # Đảm bảo bố cục gọn gàng

            canvas1 = FigureCanvasTkAgg(fig1, master=chart_container_frame)
            canvas_widget1 = canvas1.get_tk_widget()
            canvas_widget1.pack(fill=tk.BOTH, expand=True, pady=5)
            # Thêm thanh công cụ Matplotlib
            toolbar1 = NavigationToolbar2Tk(canvas1, chart_container_frame)
            toolbar1.update()
            canvas_widget1.pack(pady=2) # Điều chỉnh khoảng cách


        else:
            ttk.Label(chart_container_frame, text="Không có dữ liệu giờ làm việc để hiển thị biểu đồ.").pack(pady=10)


        # --- BIỂU ĐỒ 2: SỐ LẦN ĐI MUỘN THEO NGÀY ---
        if not metrics['late_check_in_count_by_day'].empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            metrics['late_check_in_count_by_day'].plot(kind='bar', ax=ax2, color='salmon')
            ax2.set_title('Số lần đi muộn mỗi ngày', fontsize=10)
            ax2.set_xlabel('Ngày', fontsize=8)
            ax2.set_ylabel('Số lần', fontsize=8)
            ax2.tick_params(axis='x', rotation=45, labelsize=7)
            ax2.tick_params(axis='y', labelsize=7)
            plt.tight_layout()

            canvas2 = FigureCanvasTkAgg(fig2, master=chart_container_frame)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.pack(fill=tk.BOTH, expand=True, pady=5)
            toolbar2 = NavigationToolbar2Tk(canvas2, chart_container_frame)
            toolbar2.update()
            canvas_widget2.pack(pady=2) # Điều chỉnh khoảng cách
        else:
            ttk.Label(chart_container_frame, text="Không có dữ liệu đi muộn để hiển thị biểu đồ.").pack(pady=10)

        
               


    def _on_user_tree_select(self, event):
        # Phương thức này được gọi mỗi khi có sự thay đổi lựa chọn trong Treeview
        # Bạn có thể thêm code debug ở đây nếu muốn xem sự kiện có kích hoạt hay không
        # Ví dụ:
        # selected_items = self.user_tree.selection()
        # print(f"DEBUG: Selection event triggered. Current selection: {selected_items}")
        pass # Hiện tại không cần làm gì cụ thể ở đây, chỉ cần đảm bảo sự kiện được xử lý 


    def _calculate_attendance_metrics(self, df_user_attendance):
        """Tính toán các chỉ số chấm công cần thiết cho biểu đồ."""
        metrics = {}

        df_temp = df_user_attendance.copy() 

        # Chuyển đổi các cột liên quan sang kiểu datetime object
        df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'], errors='coerce')
        df_temp['CheckInTime'] = pd.to_datetime(df_temp['CheckInTime'], errors='coerce')
        df_temp['CheckOutTime'] = pd.to_datetime(df_temp['CheckOutTime'], errors='coerce')

        # Gỡ bỏ các dòng mà việc chuyển đổi datetime bị lỗi (NaT) cho các cột thiết yếu
        df_temp.dropna(subset=['Timestamp', 'CheckInTime', 'CheckOutTime'], inplace=True)

    # --- 1. Tổng giờ làm việc hàng ngày ---
        completed_shifts = df_temp[
            (df_temp['CheckType'] == 'Check-in') & 
            (df_temp['CheckInTime'].notna()) & 
            (df_temp['CheckOutTime'].notna())
        ].copy() 

        if not completed_shifts.empty:
            completed_shifts['WorkDuration'] = completed_shifts['CheckOutTime'] - completed_shifts['CheckInTime']
            completed_shifts['WorkDuration_minutes'] = completed_shifts['WorkDuration'].dt.total_seconds() / 60

            daily_work_hours = completed_shifts.groupby(completed_shifts['Timestamp'].dt.date)['WorkDuration_minutes'].sum() / 60
            metrics['daily_work_hours'] = daily_work_hours
        else:
            metrics['daily_work_hours'] = pd.Series(dtype=float) 

    # --- 2. Số lần đi muộn theo ngày (ví dụ: nếu check-in sau 8:00 AM) ---
        start_work_time_threshold = datetime.time(8, 0, 0) # Ví dụ: 8:00 AM

        check_in_records = df_temp[
            (df_temp['CheckType'] == 'Check-in') & 
            (df_temp['CheckInTime'].notna())
        ].copy() # Đã sửa lỗi thụt lề và vị trí dấu chấm

        if not check_in_records.empty:
            late_check_ins_records = check_in_records[
                check_in_records['CheckInTime'].dt.time > start_work_time_threshold
            ]
            metrics['late_check_in_count_by_day'] = late_check_ins_records.groupby(late_check_ins_records['Timestamp'].dt.date).size()
        else:
            metrics['late_check_in_count_by_day'] = pd.Series(dtype=int) # Đã sửa lỗi thụt lề

        return metrics