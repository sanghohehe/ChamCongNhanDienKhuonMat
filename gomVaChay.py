import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox,filedialog
from PIL import Image, ImageTk
import pandas as pd
from report_window import ReportWindow



from database.database_manager import record_attendance, load_attendance, save_attendance




class FaceRecognitionApp:
   def __init__(self, root, main_frame=None):
       self.root = root
       self.root.title("Hệ thống nhận dạng khuôn mặt")


       os.makedirs('dataset', exist_ok=True)
       os.makedirs('trainer', exist_ok=True)


       self.user_identifier = ""
       self.count = 0
       self.cam = None
       self.is_capturing = False
       self.is_recognizing = False
       self.check_in = True 
       self.names = {}
       self.is_recognizing = False  # ✅ Ngăn gọi lại liên tục
       self.id_mapping = {}
       self.load_existing_users()


       self.create_widgets()
       self.root.protocol("WM_DELETE_WINDOW", self.on_close)


       self.status_label = ttk.Label(main_frame, text="Sẵn sàng", font=("Helvetica", 11), foreground="blue")
       self.status_label.pack(pady=5)


   def load_existing_users(self):
       for filename in os.listdir('dataset'):
           if filename.startswith('User.') and filename.endswith('.jpg'):
               parts = filename.split('.')
               if len(parts) >= 3:
                   user_id = parts[1]
                   name = self.get_name_for_id(user_id)
                   if name:
                       self.names[user_id] = name


       for filename in os.listdir('dataset'):
           if filename.endswith('.name.txt'):
               user_id = filename.split('.')[1]
               name = self.get_name_for_id(user_id)
               if name:
                   self.names[user_id] = name


       if os.path.exists('trainer/id_mapping.txt'):
           with open('trainer/id_mapping.txt', 'r') as f:
               for line in f.readlines():
                   parts = line.strip().split(':')
                   if len(parts) == 2:
                       numeric_id, original_id = parts
                       try:
                           num_id = int(numeric_id)
                           self.id_mapping[num_id] = original_id
                       except ValueError:
                           self.id_mapping[numeric_id] = original_id


   def get_name_for_id(self, user_id):
       name_file = f"dataset/User.{user_id}.name.txt"
       if os.path.exists(name_file):
           with open(name_file, 'r') as f:
               return f.read().strip()
       return None


   def save_name_for_id(self, user_id, name):
       name_file = f"dataset/User.{user_id}.name.txt"
       with open(name_file, 'w') as f:
           f.write(name)


   def create_widgets(self, button_frame=None, buttons=None):
       if buttons is None:
           buttons = []  # Đặt giá trị mặc định là danh sách rỗng nếu buttons không được truyền vào


       main_frame = ttk.Frame(self.root, padding="10")
       main_frame.pack(fill=tk.BOTH, expand=True)


       self.video_label = ttk.Label(main_frame)
       self.video_label.pack()


       # Nút hàng 1
       top_button_frame = ttk.Frame(main_frame)
       top_button_frame.pack(pady=10)


       btn1 = tk.Button(top_button_frame, text="📸 Chụp ảnh", command=self.start_capture,
                        bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
       btn1.pack(side=tk.LEFT, padx=15)


       btn2 = tk.Button(top_button_frame, text="🧠 Train Model", command=self.train_model,
                        bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
       btn2.pack(side=tk.LEFT, padx=15)


       # Nút hàng 2
       bottom_button_frame = ttk.Frame(main_frame)
       bottom_button_frame.pack(pady=10)


       btn3 = tk.Button(bottom_button_frame, text="🔍 Chấm Công", command=self.start_recognition,
                        bg="#FF9800", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
       btn3.pack(side=tk.LEFT, padx=15)


       btn4 = tk.Button(bottom_button_frame, text="📊 Xem Chấm Công", command=self.view_attendance,
                        bg="#9C27B0", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
       btn4.pack(side=tk.LEFT, padx=15)

       
        

    # Thêm nút Check-Out
       btn5 = tk.Button(bottom_button_frame, text="🔓 Check-Out", command=self.on_check_out_click,
                 bg="#E91E63", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
       btn5.pack(side=tk.LEFT, padx=15)



       btn6 = tk.Button(bottom_button_frame, text="📈 Thống kê - Báo cáo", command=lambda: ReportWindow(self.root),
                 bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
       btn6.pack(side=tk.LEFT, padx=15)
 


       # Tạo và thêm các nút
       for (text, command, color) in buttons:
           btn = tk.Button(button_frame, text=text, command=command, bg=color, fg="white",
                           font=("Helvetica", 10, "bold"), width=15)
           btn.pack(side=tk.LEFT, padx=10)


       self.result_label = ttk.Label(main_frame, text="", font=('Helvetica', 12))
       self.result_label.pack(fill=tk.X, pady=5)


   def start_capture(self):
       if self.is_capturing or self.is_recognizing:
           return


       self.get_user_info()
       if not self.user_identifier:
           return


       self.cam = cv2.VideoCapture(0)
       self.cam.set(3, 640)
       self.cam.set(4, 480)


       self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


       self.is_capturing = True
       self.count = 0
       self.status_label.config(text="Đang chụp ảnh... Hãy nhìn vào camera")
       self.update_capture()


   def update_capture(self):
       if not self.is_capturing:
           return


       ret, img = self.cam.read()
       if ret:
           img = cv2.flip(img, 1)
           gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           faces = self.face_detector.detectMultiScale(gray, 1.3, 5)


           for (x, y, w, h) in faces:
               cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
               self.count += 1


               save_path = f"dataset/User.{self.user_identifier}.{self.count}.jpg"
               cv2.imwrite(save_path, gray[y:y + h, x:x + w])


               cv2.putText(img, str(self.count), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
               self.result_label.config(text=f"Đã lưu {self.count} ảnh cho {self.user_identifier}")


           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           img = Image.fromarray(img)
           img = ImageTk.PhotoImage(image=img)


           self.video_label.img = img
           self.video_label.config(image=img)


           if self.count >= 200:
               self.stop_all()
               messagebox.showinfo("Thành công", "Đã chụp đủ 200 ảnh")
               return


       self.root.after(10, self.update_capture)


   def train_model(self):
       if self.is_capturing or self.is_recognizing:
           return


       path = 'dataset'
       recognizer = cv2.face.LBPHFaceRecognizer_create()
       detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


       def getImagesAndLabels(path):
           imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
           faceSamples = []
           ids = []


           for imagePath in imagePaths:
               try:
                   filename = os.path.split(imagePath)[-1]
                   parts = filename.split('.')
                   if len(parts) >= 3:
                       id_part = parts[1]


                       PIL_img = Image.open(imagePath).convert('L')
                       img_numpy = np.array(PIL_img, 'uint8')
                       faces = detector.detectMultiScale(img_numpy)


                       for (x, y, w, h) in faces:
                           faceSamples.append(img_numpy[y:y + h, x:x + w])
                           ids.append(id_part)


                       if id_part not in self.names:
                           self.names[id_part] = id_part
                           self.save_name_for_id(id_part, id_part)
               except Exception as e:
                   print(f"Lỗi khi xử lý {imagePath}: {str(e)}")


           return faceSamples, ids


       self.status_label.config(text="Đang train model... Vui lòng chờ")
       self.root.update()


       try:
           faces, original_ids = getImagesAndLabels(path)
           if len(faces) == 0:
               messagebox.showerror("Lỗi", "Không tìm thấy khuôn mặt nào")
               return


           unique_ids = list(set(original_ids))
           self.id_mapping = {i + 1: id for i, id in enumerate(unique_ids)}
           numeric_ids = [list(self.id_mapping.keys())[list(self.id_mapping.values()).index(id)] for id in original_ids]


           recognizer.train(faces, np.array(numeric_ids))
           recognizer.write('trainer/trainer.yml')


           with open('trainer/id_mapping.txt', 'w') as f:
               for numeric_id, original_id in self.id_mapping.items():
                   f.write(f"{numeric_id}:{original_id}\n")


           messagebox.showinfo("Thành công", f"Train hoàn tất. Đã train {len(unique_ids)} khuôn mặt.")
       except Exception as e:
           messagebox.showerror("Lỗi", f"Train thất bại: {str(e)}")


       self.status_label.config(text="Train hoàn thành")


   def start_recognition(self):
       self.has_checked = False       # reset cờ
       self.recognition_start_time = time.time()
       if self.is_capturing or self.is_recognizing:
           return

       try:
           self.recognizer = cv2.face.LBPHFaceRecognizer_create()
           self.recognizer.read('trainer/trainer.yml')
           self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
           self.font = cv2.FONT_HERSHEY_SIMPLEX


           self.id_mapping = {}
           if os.path.exists('trainer/id_mapping.txt'):
               with open('trainer/id_mapping.txt', 'r') as f:
                   for line in f.readlines():
                       parts = line.strip().split(':')
                       if len(parts) == 2:
                           numeric_id, original_id = parts
                           self.id_mapping[int(numeric_id)] = original_id
       except Exception as e:
           messagebox.showerror("Lỗi", f"Không thể tải model: {str(e)}")
           return


       self.cam = cv2.VideoCapture(0)
       self.cam.set(3, 640)
       self.cam.set(4, 480)


       self.minW = 0.1 * self.cam.get(3)
       self.minH = 0.1 * self.cam.get(4)


       self.is_recognizing = True
       self.status_label.config(text="Đang nhận dạng...")
       self.update_recognition()


   def update_recognition(self):
    if not self.is_recognizing:
        return

    ret, img = self.cam.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray, 1.2, 5,
            minSize=(int(self.minW), int(self.minH))
        )

        for (x, y, w, h) in faces:
            numeric_id, confidence = self.recognizer.predict(
                gray[y:y + h, x:x + w]
            )
            confidence_percent = round(100 - confidence)
            original_id = self.id_mapping.get(
                numeric_id, str(numeric_id)
            )
            name = self.names.get(original_id, original_id)

            if confidence_percent > 50 and not self.has_checked:
                # 1) lần đầu tiên gặp face → ghi attendance
                record_attendance(original_id, name, check_in=self.check_in)
                self.has_checked = True    # chặn ghi thêm

                # 2) dừng ngay lập tức
                self.is_recognizing = False
                self.stop_all()
                messagebox.showinfo("Thành công", f"Đã nhận diện: {name}")
                return

            # hiển thị bounding box & label
            color = (0,255,0) if confidence_percent>50 else (0,0,255)
            text = f"{name} ({confidence_percent}%)" if confidence_percent>50 else "Unknown"
            cv2.putText(img, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Cập nhật hình ảnh camera lên GUI
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.img = img
        self.video_label.config(image=img)

    # Nếu vẫn đang nhận dạng thì tiếp tục lặp
    if self.is_recognizing:
        self.root.after(10, self.update_recognition)




#    def on_check_in_click(self):
#     self.check_in = True
#     self.start_recognition()

   def on_check_out_click(self):
    self.check_in = False
    self.start_recognition()
   


   def stop_all(self):
       self.is_capturing = False
       self.is_recognizing = False
       if self.cam:
           self.cam.release()
           self.cam = None
       self.status_label.config(text="Sẵn sàng")
       self.video_label.config(image='')


   def get_user_info(self):
       input_window = tk.Toplevel(self.root)
       input_window.title("Nhập thông tin người dùng")


       ttk.Label(input_window, text="Nhập ID:").grid(row=0, column=0, padx=5, pady=5)
       id_entry = ttk.Entry(input_window)
       id_entry.grid(row=0, column=1, padx=5, pady=5)


       ttk.Label(input_window, text="Nhập Tên:").grid(row=1, column=0, padx=5, pady=5)
       name_entry = ttk.Entry(input_window)
       name_entry.grid(row=1, column=1, padx=5, pady=5)


       def save_info():
           user_id = id_entry.get().strip()
           name = name_entry.get().strip()
           if not user_id or not name:
               messagebox.showerror("Lỗi", "Vui lòng nhập đầy đủ ID và Tên")
               return


           self.user_identifier = user_id
           self.names[user_id] = name
           self.save_name_for_id(user_id, name)
           input_window.destroy()


       ttk.Button(input_window, text="Xác nhận", command=save_info).grid(row=2, column=0, columnspan=2, pady=10)


       input_window.transient(self.root)
       input_window.grab_set()
       self.root.wait_window(input_window)


    

    


   def view_attendance(self):
    window = tk.Toplevel(self.root)
    window.title("Quản lý Chấm Công")

    self.attendance_labels = []  # Reset mỗi lần mở

    # --- Bộ lọc ---
    filter_frame = ttk.Frame(window)
    filter_frame.pack(pady=5)

    ttk.Label(filter_frame, text="Lọc theo ngày (YYYY-MM-DD):").grid(row=0, column=0)
    date_entry = ttk.Entry(filter_frame)
    date_entry.grid(row=0, column=1)

    ttk.Label(filter_frame, text="Lọc theo ID/Tên:").grid(row=1, column=0)
    user_entry = ttk.Entry(filter_frame)
    user_entry.grid(row=1, column=1)

    # --- Bảng dữ liệu ---
    table_frame = ttk.Frame(window)
    table_frame.pack(padx=10, pady=10)

    # --- Hàm cập nhật bảng ---
    def update_table(df):
        # Xóa các label cũ
        for label in self.attendance_labels:
            label.destroy()
        self.attendance_labels = []

        # Các tiêu đề cột đầy đủ
        headers = ["ID", "Name", "Date", "Time", "CheckIn", "CheckOut", "Status"]
        for col, header in enumerate(headers):
            lbl = ttk.Label(table_frame, text=header, font=('Helvetica', 10, 'bold'))
            lbl.grid(row=0, column=col, padx=5, pady=5)
            self.attendance_labels.append(lbl)

        # Dữ liệu từng dòng
        for i, row in df.iterrows():
            for j, header in enumerate(headers):
                value = row.get(header, "-")
                value = value if pd.notna(value) else "-"
                lbl = ttk.Label(table_frame, text=value)
                lbl.grid(row=i + 1, column=j, padx=5, pady=5)
                self.attendance_labels.append(lbl)

    # --- Lọc dữ liệu ---
    def apply_filters():
        from database.database_manager import filter_attendance_by_date, filter_attendance_by_user, load_attendance
        date = date_entry.get()
        user = user_entry.get()

        if date:
            filtered_df = filter_attendance_by_date(date)
        elif user:
            filtered_df = filter_attendance_by_user(user)
        else:
            filtered_df = load_attendance()

        update_table(filtered_df)

    # Nút lọc
    ttk.Button(filter_frame, text="Lọc", command=apply_filters).grid(row=2, column=0, columnspan=2, pady=5)

    # Nút xuất Excel
    def export_to_excel():
        from database.database_manager import load_attendance
        df = load_attendance()
        export_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if export_path:
            df.to_excel(export_path, index=False)
            messagebox.showinfo("Thành công", f"Đã xuất dữ liệu ra file:\n{export_path}")

    ttk.Button(filter_frame, text="Xuất Excel", command=export_to_excel).grid(row=3, column=0, columnspan=2, pady=5)

    # Hiển thị dữ liệu lần đầu
    from database.database_manager import load_attendance
    







    update_table(load_attendance())



    # Hàm xuất dữ liệu ra file Excel
#    def export_to_excel():
#     if not hasattr(window, 'current_df') or window.current_df.empty:
#         messagebox.showwarning("Cảnh báo", "Không có dữ liệu để xuất.")
#         return

#     file_path = filedialog.asksaveasfilename(
#         defaultextension=".xlsx",
#         filetypes=[("Excel Files", "*.xlsx")],
#         title="Lưu file chấm công"
#     )

#     if file_path:
#         try:
#             window.current_df.to_excel(file_path, index=False)
#             messagebox.showinfo("Thành công", f"Đã xuất file Excel: {file_path}")
#         except Exception as e:
#             messagebox.showerror("Lỗi", f"Không thể xuất file: {e}")
   


   def on_close(self, attendance_df=None):
       """Dừng camera, lưu lịch sử và thoát khi đóng cửa sổ"""
       try:
           self.stop_all()  # 👉 Dừng nhận diện / chụp ảnh nếu đang chạy
           save_attendance(load_attendance())  # Lưu dữ liệu chấm công (nếu cần)
       except Exception as e:
           messagebox.showerror("Lỗi", f"Không thể lưu lịch sử chấm công: {str(e)}")
       finally:
           self.root.quit()
           self.root.destroy()










def summarize_checkin_checkout(df):
       df['Date'] = df['Timestamp'].apply(lambda x: x.date())
       summary = df.groupby(['ID', 'Name', 'Date'])['Timestamp'].agg(['min', 'max']).reset_index()
       summary.columns = ['ID', 'Name', 'Date', 'Check-in', 'Check-out']
       return summary







    


if __name__ == "__main__":
   root = tk.Tk()
   app = FaceRecognitionApp(root)
   root.mainloop()







