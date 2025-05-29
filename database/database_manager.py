import pandas as pd
import os
from datetime import datetime, timedelta
import config

# Đường dẫn file
ATTENDANCE_FILE = config.ATTENDANCE_FILE # <--- SỬA DÒNG NÀY

# Đảm bảo thư mục DATA_DIR tồn tại (vì config.py đã đảm bảo điều này)
os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True) # Đảm bảo thư mục chứa ATTENDANCE_FILE tồn tại

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['UserID'] = df['UserID'].astype(str)
            # Chuyển đổi CheckInTime và CheckOutTime sang datetime (nếu có)
            if 'CheckInTime' in df.columns:
                df['CheckInTime'] = pd.to_datetime(df['CheckInTime'], errors='coerce')
            if 'CheckOutTime' in df.columns:
                df['CheckOutTime'] = pd.to_datetime(df['CheckOutTime'], errors='coerce')
            return df
        except pd.errors.EmptyDataError:
            print(f"File {ATTENDANCE_FILE} rỗng hoặc không đúng định dạng. Trả về DataFrame rỗng.")
            return pd.DataFrame(columns=['UserID', 'Name', 'Timestamp', 'CheckType', 'CheckInTime', 'CheckOutTime'])
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu chấm công từ {ATTENDANCE_FILE}: {e}")
            return pd.DataFrame(columns=['UserID', 'Name', 'Timestamp', 'CheckType', 'CheckInTime', 'CheckOutTime'])
    else:
        print(f"File {ATTENDANCE_FILE} không tồn tại. Tạo DataFrame rỗng và lưu.")
        df = pd.DataFrame(columns=['UserID', 'Name', 'Timestamp', 'CheckType', 'CheckInTime', 'CheckOutTime'])
        df.to_csv(ATTENDANCE_FILE, index=False)
        return df

def save_attendance(df=None):
    if df is None:
        df_to_save = load_attendance()
    else:
        df_to_save = df
    
    expected_cols = ['UserID', 'Name', 'Timestamp', 'CheckType', 'CheckInTime', 'CheckOutTime']
    for col in expected_cols:
        if col not in df_to_save.columns:
            df_to_save[col] = pd.NA 

    df_to_save = df_to_save[expected_cols]

    for col in ['Timestamp', 'CheckInTime', 'CheckOutTime']:
        if col in df_to_save.columns and df_to_save[col].dtype == 'datetime64[ns]':
            df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_to_save.to_csv(ATTENDANCE_FILE, index=False)
    print(f"Dữ liệu chấm công đã được lưu vào {ATTENDANCE_FILE}")

def record_attendance(user_id, name, check_type):
    df = load_attendance()
    current_time = datetime.now()
    
    last_record = df[(df['UserID'] == user_id)].sort_values(by='Timestamp', ascending=False).head(1)
    
    new_record = {
        'UserID': user_id,
        'Name': name,
        'Timestamp': current_time,
        'CheckType': check_type,
        'CheckInTime': pd.NaT,
        'CheckOutTime': pd.NaT
    }

    if check_type == "Check-in":
        new_record['CheckInTime'] = current_time
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        print(f"{name} (ID: {user_id}) đã Check-in lúc {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    elif check_type == "Check-out":
        new_record['CheckOutTime'] = current_time
        
        if not last_record.empty and pd.isna(last_record['CheckOutTime'].iloc[0]):
            df.loc[last_record.index[0], 'CheckOutTime'] = current_time
            print(f"{name} (ID: {user_id}) đã Check-out lúc {current_time.strftime('%Y-%m-%d %H:%M:%S')} (cập nhật bản ghi cũ)")
        else:
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            print(f"{name} (ID: {user_id}) đã Check-out lúc {current_time.strftime('%Y-%m-%d %H:%M:%S')} (bản ghi mới)")
            
    save_attendance(df)

def summarize_checkin_checkout(df_attendance):
    print(f"\nDEBUG_SUMMARIZE: --- Bắt đầu hàm summarize_checkin_checkout ---")
    print(f"DEBUG_SUMMARIZE: Input df_attendance shape: {df_attendance.shape}")
    print(f"DEBUG_SUMMARIZE: Input df_attendance columns: {df_attendance.columns.tolist()}")
    print(f"DEBUG_SUMMARIZE: Input df_attendance head:\n{df_attendance.head()}")

    df = df_attendance.copy()
    
    # Đảm bảo các cột thời gian là datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['CheckInTime'] = pd.to_datetime(df['CheckInTime'], errors='coerce')
    df['CheckOutTime'] = pd.to_datetime(df['CheckOutTime'], errors='coerce')
    
    # Xóa các hàng có Timestamp NaT nếu có lỗi chuyển đổi
    df.dropna(subset=['Timestamp'], inplace=True)
    
    df = df.sort_values(by=['UserID', 'Timestamp'])

    summary_data = []

    unique_users = df['UserID'].unique()
    print(f"DEBUG_SUMMARIZE: Unique users found: {unique_users}")

    if len(unique_users) == 0:
        print("DEBUG_SUMMARIZE: No unique users found in attendance data. Returning empty summary.")
        # Đảm bảo trả về DataFrame với các cột đúng để tránh lỗi hiển thị
        return pd.DataFrame(columns=['UserID', 'Name', 'TotalCheckIn', 'TotalCheckOut', 'AvgWorkDuration', 'Status']) 

    for user_id in unique_users:
        print(f"\nDEBUG_SUMMARIZE: --- Đang xử lý người dùng: {user_id} ---")
        user_df = df[df['UserID'] == user_id].copy() # Dùng .copy() để tránh SettingWithCopyWarning
        
        print(f"DEBUG_SUMMARIZE: user_df for {user_id} shape: {user_df.shape}")
        print(f"DEBUG_SUMMARIZE: user_df for {user_id} head:\n{user_df.head()}")
        
        if user_df.empty:
            print(f"DEBUG_SUMMARIZE: user_df is empty for {user_id}. Skipping.")
            continue # Bỏ qua người dùng này nếu không có dữ liệu

        name = user_df['Name'].iloc[-1] if not user_df.empty else "Unknown" 
        print(f"DEBUG_SUMMARIZE: User Name: {name}")

        # Tính tổng số lượt Check-in và Check-out thực tế
        total_check_in = user_df['CheckInTime'].count() # Đếm số bản ghi có giá trị CheckInTime
        total_check_out = user_df['CheckOutTime'].count() # Đếm số bản ghi có giá trị CheckOutTime
        print(f"DEBUG_SUMMARIZE: Calculated Total Check-in: {total_check_in}, Total Check-out: {total_check_out}")

        total_work_duration = timedelta(0)
        valid_pairs_count = 0
        
        # Lọc các bản ghi có ít nhất một trong hai CheckInTime/CheckOutTime để tính duration
        df_filtered_for_duration = user_df.dropna(subset=['CheckInTime', 'CheckOutTime'], how='all')
        print(f"DEBUG_SUMMARIZE: df_filtered_for_duration shape: {df_filtered_for_duration.shape}")
        
        current_check_in_time = None
        if not df_filtered_for_duration.empty:
            df_pairs = df_filtered_for_duration.sort_values(by='Timestamp')
            for index, row in df_pairs.iterrows():
                # Nếu bản ghi có CheckInTime, đây là điểm bắt đầu tiềm năng của một ca làm việc
                if pd.notna(row['CheckInTime']):
                    current_check_in_time = row['CheckInTime']
                
                # Nếu bản ghi có CheckOutTime VÀ có current_check_in_time hợp lệ
                if pd.notna(row['CheckOutTime']) and current_check_in_time is not None:
                    if row['CheckOutTime'] > current_check_in_time:
                        duration = row['CheckOutTime'] - current_check_in_time
                        
                        # Giới hạn thời gian làm việc tối đa theo config
                        if duration <= timedelta(seconds=config.MAX_WORK_SESSION_HOURS):
                            total_work_duration += duration
                            valid_pairs_count += 1
                        else:
                            print(f"DEBUG_SUMMARIZE: Duration for {user_id} exceeded MAX_WORK_SESSION_HOURS: {duration}")
                    else:
                        print(f"DEBUG_SUMMARIZE: CheckOutTime before CheckInTime for {user_id}. Skipping pair.")
                    current_check_in_time = None # Reset cho cặp tiếp theo, dù có hợp lệ hay không hợp lệ về thời gian
                # else: Nếu không có CheckOutTime hoặc current_check_in_time là None, chỉ cần bỏ qua

        print(f"DEBUG_SUMMARIZE: Valid pairs count: {valid_pairs_count}, Total work duration: {total_work_duration}")

        avg_work_duration = total_work_duration / valid_pairs_count if valid_pairs_count > 0 else timedelta(0)
        print(f"DEBUG_SUMMARIZE: Avg work duration: {avg_work_duration}")

        status = "Chưa có dữ liệu chấm công" # Mặc định
        
        if not user_df.empty:
            latest_record = user_df.sort_values(by='Timestamp', ascending=False).iloc[0]
            print(f"DEBUG_SUMMARIZE: Latest record for status check:\n{latest_record}")
            
            is_check_in_recorded = pd.notna(latest_record['CheckInTime'])
            is_check_out_recorded = pd.notna(latest_record['CheckOutTime'])
            
            print(f"DEBUG_SUMMARIZE: is_check_in_recorded: {is_check_in_recorded}, is_check_out_recorded: {is_check_out_recorded}")

            if is_check_in_recorded and is_check_out_recorded:
                status = f"Đã về (Check-out lúc {latest_record['CheckOutTime'].strftime('%H:%M')})"
            elif is_check_in_recorded and not is_check_out_recorded:
                status = f"Đang làm việc (Check-in lúc {latest_record['CheckInTime'].strftime('%H:%M')})"
            elif not is_check_in_recorded and is_check_out_recorded:
                status = f"Chỉ Check-out (lúc {latest_record['CheckOutTime'].strftime('%H:%M')})"
            # else: status vẫn là "Chưa có dữ liệu chấm công" nếu cả hai đều NaT
        
        print(f"DEBUG_SUMMARIZE: Final Status for {user_id}: {status}")

        summary_data.append({
            'UserID': user_id,
            'Name': name,
            'TotalCheckIn': total_check_in,
            'TotalCheckOut': total_check_out,
            'AvgWorkDuration': str(avg_work_duration).split('.')[0] if valid_pairs_count > 0 else '-', 
            'Status': status,
        })
    
    final_summary = pd.DataFrame(summary_data)
    print(f"\nDEBUG_SUMMARIZE: --- Hàm summarize_checkin_checkout kết thúc ---")
    print(f"DEBUG_SUMMARIZE: Final summary DataFrame shape: {final_summary.shape}")
    print(f"DEBUG_SUMMARIZE: Final summary DataFrame head:\n{final_summary.head()}")
    return final_summary

def update_user_name_in_attendance(user_id, new_name):
    """
    Cập nhật trường 'Name' trong attendance.csv cho một UserID cụ thể.
    """
    df = load_attendance()
    if not df.empty and user_id in df['UserID'].values:
        # Chỉ cập nhật các bản ghi có UserID khớp
        df.loc[df['UserID'] == user_id, 'Name'] = new_name
        save_attendance(df)
        print(f"Đã cập nhật tên '{new_name}' cho UserID '{user_id}' trong attendance.csv.")
    else:
        print(f"Không tìm thấy UserID '{user_id}' trong attendance.csv để cập nhật tên.")
