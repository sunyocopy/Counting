from flask import Flask, request, Response
import cv2
import numpy as np
from ultralytics import YOLO

# สร้าง Flask app
app = Flask(__name__)

# โหลดโมเดล YOLOv8 ของคุณ
model = YOLO('steel.pt')

# สร้าง Endpoint สำหรับ API
@app.route('/predict', methods=['POST'])
def predict():
    # รับไฟล์รูปภาพจาก request
    file = request.files['image']
    img_bytes = file.read()

    # แปลงข้อมูลรูปภาพเป็น format ที่ OpenCV ใช้ได้
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ส่งภาพให้โมเดลประมวลผล
    results = model(frame, conf=0.7)

    # วาดผลลัพธ์ลงบนภาพ
    annotated_frame = results[0].plot(font_size=0.5, line_width=2)

    # แปลงภาพผลลัพธ์กลับเป็นไฟล์ JPG เพื่อส่งกลับไป
    _, buffer = cv2.imencode('.jpg', annotated_frame)

    # ส่งภาพผลลัพธ์กลับไปเป็น response
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)