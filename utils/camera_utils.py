import cv2
import os

def initialize_camera(camera_index=0):
    """
    Khởi tạo và trả về đối tượng camera.
    Sử dụng camera_index=0 cho camera mặc định.
    """
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print(f"Lỗi: Không thể mở camera với index {camera_index}.")
        print("Vui lòng kiểm tra:")
        print("1. Camera có đang được sử dụng bởi ứng dụng khác không?")
        print("2. Driver camera đã được cài đặt và cập nhật chưa?")
        print("3. Index camera có đúng không (thử 0, 1, ...)?")
        return None
    print(f"Camera với index {camera_index} đã được mở thành công.")
    return cam

def release_camera(camera):
    """Giải phóng đối tượng camera."""
    if camera and camera.isOpened():
        camera.release()
        print("Camera đã được giải phóng.")

def load_face_detector():
    """Tải bộ phát hiện khuôn mặt Haar Cascade."""
    # Đảm bảo đường dẫn đến file Haar Cascade là chính xác
    # Tên file thường là haarcascade_frontalface_default.xml
    face_detector_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(face_detector_path):
        # Nếu không tìm thấy trong thư mục mặc định của OpenCV,
        # kiểm tra xem bạn có đặt nó ở một vị trí khác (ví dụ: thư mục 'cascades' trong project của bạn)
        # Đảm bảo bạn đã tải file này và đặt đúng chỗ.
        print(f"Lỗi: Không tìm thấy file phát hiện khuôn mặt tại {face_detector_path}")
        # Bạn có thể thử một đường dẫn khác nếu bạn đã đặt nó ở nơi khác, ví dụ:
        # face_detector_path = "cascades/haarcascade_frontalface_default.xml"
        # if not os.path.exists(face_detector_path):
        #     raise FileNotFoundError("Không tìm thấy file phát hiện khuôn mặt. Vui lòng tải và đặt nó đúng chỗ.")
        raise FileNotFoundError(f"Không tìm thấy file phát hiện khuôn mặt: {face_detector_path}. Vui lòng kiểm tra lại đường dẫn hoặc cài đặt OpenCV.")


    faceCascade = cv2.CascadeClassifier(face_detector_path)
    if faceCascade.empty():
        raise IOError(f"Không thể tải bộ phân loại Haar Cascade từ {face_detector_path}.")
    return faceCascade