import csv
import os
from datetime import datetime
import pandas as pd

ATTENDANCE_FILE = 'data/attendance.csv'

def ensure_file_exists():
    try:
        os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)  # Tạo thư mục nếu chưa có
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Name", "Date", "Time", "CheckIn", "CheckOut", "Status"])  # Đã sửa: thêm Status
    except Exception as e:
        print(f"Lỗi khi đảm bảo tệp tin tồn tại: {str(e)}")

def record_attendance(user_id, name, check_in=True):
    try:
        ensure_file_exists()

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H:%M:%S')

        df = load_attendance()

        # Nếu check-in
        if check_in:
            # Kiểm tra đã check-in chưa
            existing = df[(df['ID'] == int(user_id)) & (df['Date'] == date_str) & (df['Status'] == 'In')]
            if not existing.empty:
                print(f"{name} đã check-in rồi hôm nay.")
                return

            new_record = pd.DataFrame([[user_id, name, date_str, time_str, timestamp_str, '', '', '', 'In']],
                                      columns=['ID', 'Name', 'Date', 'Time', 'CheckIn', 'CheckOut', 'Notes', 'Timestamp', 'Status'])
            df = pd.concat([df, new_record], ignore_index=True)
            print(f"Đã check-in: {name}")

        else:  # Nếu là check-out
            # Tìm dòng check-in gần nhất chưa check-out
            idx = df[(df['ID'] == int(user_id)) & (df['Date'] == date_str) & (df['Status'] == 'In')].last_valid_index()

            if idx is None:
                print(f"[Check Out] Không thể chấm công ra vì chưa chấm công vào hoặc đã chấm công ra.")
                return

            df.at[idx, 'CheckOut'] = timestamp_str
            df.at[idx, 'Status'] = 'Out'
            print(f"Đã check-out: {name}")

        df.to_csv(ATTENDANCE_FILE, index=False)

    except Exception as e:
        print(f"Lỗi khi ghi dữ liệu chấm công: {str(e)}")

    

def load_attendance():
    try:
        ensure_file_exists()
        df = pd.read_csv(ATTENDANCE_FILE)
        
        # Kiểm tra dữ liệu đã được tải chính xác chưa
        print(f"Dữ liệu chấm công đã được tải: {df.head()}")
        
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu chấm công: {str(e)}")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu có lỗi

def filter_attendance_by_date(date):
    try:
        df = load_attendance()
        return df[df['Date'] == date]
    except Exception as e:
        print(f"Lỗi khi lọc dữ liệu theo ngày: {str(e)}")
        return pd.DataFrame()

def filter_attendance_by_user(user):
    try:
        df = load_attendance()
        return df[(df['ID'] == user) | (df['Name'].str.lower() == user.lower())]
    except Exception as e:
        print(f"Lỗi khi lọc dữ liệu theo người dùng: {str(e)}")
        return pd.DataFrame()

def delete_attendance(index):
    try:
        df = load_attendance()
        df = df.drop(index)
        df.to_csv(ATTENDANCE_FILE, index=False)
    except Exception as e:
        print(f"Lỗi khi xóa bản ghi chấm công: {str(e)}")

def save_attendance(df):
    try:
        df.to_csv(ATTENDANCE_FILE, index=False)
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu chấm công: {str(e)}")
