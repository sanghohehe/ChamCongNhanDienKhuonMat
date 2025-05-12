# report_window.py
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class ReportWindow:
    def __init__(self, root):
        self.root = tk.Toplevel(root)
        self.root.title("Thống kê - Báo cáo")
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        ttk.Label(frame, text="Chọn tháng (YYYY-MM):").grid(row=0, column=0, sticky=tk.W)
        self.month_entry = ttk.Entry(frame)
        self.month_entry.grid(row=0, column=1)

        ttk.Button(frame, text="Xem Thống Kê", command=self.show_report).grid(row=0, column=2, padx=10)

        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def show_report(self):
      month = self.month_entry.get()
      try:
        df = pd.read_csv("data/attendance.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df['CheckIn'] = pd.to_datetime(df['CheckIn'])
        df['CheckOut'] = pd.to_datetime(df['CheckOut'])
        df['WorkedHours'] = (df['CheckOut'] - df['CheckIn']).dt.total_seconds() / 3600

        if month:
            df = df[df['Date'].dt.strftime('%Y-%m') == month]

        if df.empty:
            messagebox.showinfo("Thông báo", "Không có dữ liệu cho tháng này.")
            return

        hours_per_day = df.groupby(df['Date'].dt.date)['WorkedHours'].sum()
        late_count = sum(df['CheckIn'].dt.time > datetime.strptime("08:00", "%H:%M").time())
        early_leave_count = sum(df['CheckOut'].dt.time < datetime.strptime("17:00", "%H:%M").time())

        self.plot_charts(hours_per_day, late_count, early_leave_count)

        # 👇 Thêm đoạn này để liệt kê ai đi muộn / về sớm
        late_users = df[df['CheckIn'].dt.time > datetime.strptime("08:00", "%H:%M").time()]
        early_users = df[df['CheckOut'].dt.time < datetime.strptime("17:00", "%H:%M").time()]

        late_list = late_users['Name'].tolist() if 'Name' in df.columns else []
        early_list = early_users['Name'].tolist() if 'Name' in df.columns else []

        info = "📌 Danh sách đi muộn:\n"
        info += '\n'.join(late_list) if late_list else "✅ Không ai đi muộn."
        info += "\n\n📌 Danh sách về sớm:\n"
        info += '\n'.join(early_list) if early_list else "✅ Không ai về sớm."

        messagebox.showinfo("Chi tiết đi muộn / về sớm", info)

      except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xử lý dữ liệu: {e}")


    def plot_charts(self, hours_per_day, late_count, early_leave_count):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].bar(hours_per_day.index.astype(str), hours_per_day.values, color='skyblue')
        axs[0].set_title("Tổng giờ làm theo ngày")
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].set_ylabel("Giờ")

        axs[1].bar(['Đi muộn', 'Về sớm'], [late_count, early_leave_count], color=['orange', 'red'])
        axs[1].set_title("Số lần đi muộn / về sớm")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
