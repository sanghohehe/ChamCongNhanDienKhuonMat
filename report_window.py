import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from datetime import datetime
import os

# Import các hàm từ database_manager
from database.database_manager import load_attendance, summarize_checkin_checkout

class ReportWindow:
    def __init__(self, master):
        self.master = master
        self.report_window = tk.Toplevel(master)
        self.report_window.title("Báo cáo Chấm công Tổng hợp")
        self.report_window.geometry("1000x700")

        self.report_window.grab_set() # Chặn tương tác với cửa sổ chính
        self.report_window.transient(master) # Đặt cửa sổ chính là cha

        # Khung chứa Treeview và Scrollbar
        tree_frame = ttk.Frame(self.report_window, padding="10")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.report_tree = ttk.Treeview(tree_frame)
        self.report_tree.pack(fill=tk.BOTH, expand=True)

        # Cấu hình Treeview
        cols = ('ID', 'Name', 'TotalCheckIn', 'TotalCheckOut', 'AvgWorkDuration', 'Status')
        self.report_tree['columns'] = cols
        self.report_tree.column("#0", width=0, stretch=tk.NO)
        self.report_tree.column("ID", anchor=tk.CENTER, width=80)
        self.report_tree.column("Name", anchor=tk.W, width=150)
        self.report_tree.column("TotalCheckIn", anchor=tk.CENTER, width=100)
        self.report_tree.column("TotalCheckOut", anchor=tk.CENTER, width=100)
        self.report_tree.column("AvgWorkDuration", anchor=tk.CENTER, width=150)
        self.report_tree.column("Status", anchor=tk.W, width=200)

        self.report_tree.heading("#0", text="")
        self.report_tree.heading("ID", text="Mã NV")
        self.report_tree.heading("Name", text="Tên")
        self.report_tree.heading("TotalCheckIn", text="Tổng Check-in")
        self.report_tree.heading("TotalCheckOut", text="Tổng Check-out")
        self.report_tree.heading("AvgWorkDuration", text="TG làm việc TB")
        self.report_tree.heading("Status", text="Trạng thái")

        # Scrollbar cho Treeview
        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.report_tree.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_tree.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.report_tree.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.report_tree.configure(xscrollcommand=scrollbar_x.set)

        # Nút xuất Excel
        export_button_frame = ttk.Frame(self.report_window, padding="10")
        export_button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Button(export_button_frame, text="Xuất báo cáo Excel", command=self._export_report_to_excel).pack(pady=5)
        
        # ✅ Tự động tạo báo cáo khi cửa sổ mở
        self._generate_report()

    def _generate_report(self):
        """Tạo báo cáo tổng hợp từ toàn bộ dữ liệu chấm công."""
        for i in self.report_tree.get_children():
            self.report_tree.delete(i) # Xóa dữ liệu cũ

        df_attendance = load_attendance()
        if df_attendance.empty:
            messagebox.showinfo("Thông báo", "Không có dữ liệu chấm công để tạo báo cáo.")
            return

        # Gọi hàm summarize_checkin_checkout từ database_manager
        # Giả định summarize_checkin_checkout đã được sửa để hoạt động với toàn bộ df
        df_summary = summarize_checkin_checkout(df_attendance)

        if df_summary.empty:
            messagebox.showinfo("Thông báo", "Không có dữ liệu tổng hợp sau khi xử lý.")
            return

        for index, row in df_summary.iterrows():
            avg_duration_str = str(row['AvgWorkDuration']).split(' days')[-1].strip() if pd.notna(row['AvgWorkDuration']) else '-'
            
            self.report_tree.insert("", "end", values=(
                row['UserID'],
                row['Name'],
                row['TotalCheckIn'],
                row['TotalCheckOut'],
                avg_duration_str,
                row['Status']
            ))
        messagebox.showinfo("Báo cáo", f"Đã tạo báo cáo tổng hợp cho {len(df_summary)} người dùng.")

    def _export_report_to_excel(self):
        """Xuất báo cáo hiện tại ra file Excel."""
        df_attendance = load_attendance()
        if df_attendance.empty:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu để xuất.")
            return
        
        df_summary = summarize_checkin_checkout(df_attendance)

        if df_summary.empty:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu tổng hợp để xuất.")
            return

        # Làm sạch cột thời gian để xuất Excel dễ hơn
        df_summary_export = df_summary.copy()
        df_summary_export['AvgWorkDuration'] = df_summary_export['AvgWorkDuration'].apply(
            lambda x: str(x).split(' days')[-1].strip() if pd.notna(x) else '-'
        )

        df_summary_export = df_summary_export.rename(columns={
            'UserID': 'Mã Nhân Viên',
            'Name': 'Tên Nhân Viên',
            'TotalCheckIn': 'Tổng số lần Check-in',
            'TotalCheckOut': 'Tổng số lần Check-out',
            'AvgWorkDuration': 'Thời gian làm việc trung bình',
            'Status': 'Trạng thái chung' # Ví dụ: "Đi làm đầy đủ", "Thường xuyên đi muộn", v.v.
        })
        
        cols_to_export = ['Mã Nhân Viên', 'Tên Nhân Viên', 'Tổng số lần Check-in', 
                          'Tổng số lần Check-out', 'Thời gian làm việc trung bình', 'Trạng thái chung']
        df_final_export = df_summary_export[cols_to_export]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Lưu báo cáo tổng hợp chấm công"
        )

        if file_path:
            try:
                df_final_export.to_excel(file_path, index=False)
                messagebox.showinfo("Thành công", f"Đã xuất báo cáo thành công tại:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xuất báo cáo ra Excel: {str(e)}")