from flask import Flask, request, Response
import cv2
import numpy as np
from ultralytics import YOLO

# สร้าง Flask app
app = Flask(__name__)

# โหลดโมเดล YOLOv8 ของคุณ
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    # ในกรณีนี้ อาจจะจบการทำงานไปเลยถ้าโมเดลคือหัวใจหลัก
    # แต่เราจะปล่อยให้ API ทำงานต่อ เผื่อการดีบัก
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return Response("Error: Model could not be loaded.", status=500, mimetype='text/plain')

    try:
        # รับไฟล์รูปภาพจาก request
        if 'image' not in request.files:
            return Response("Error: No image file found in request.", status=400, mimetype='text/plain')
            
        file = request.files['image']
        img_bytes = file.read()
        
        # แปลงข้อมูลรูปภาพเป็น format ที่ OpenCV ใช้ได้
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- ส่วนแก้ไขปัญหา RAM หมด ---
        # ย่อขนาดรูปภาพให้มีความกว้างไม่เกิน 800px เพื่อประหยัด RAM
        new_width = 800
        height, width, _ = frame.shape
        if width > new_width:
            aspect_ratio = height / width
            new_height = int(new_width * aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))
        else:
            resized_frame = frame
        # ---------------------------

        # ส่งภาพที่ย่อขนาดแล้วให้โมเดลประมวลผล
        results = model(resized_frame, conf=0.7)

        # วาดผลลัพธ์ลงบนภาพ
        annotated_frame = results[0].plot()

        # แปลงภาพผลลัพธ์กลับเป็นไฟล์ JPG เพื่อส่งกลับไป
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        
        # ส่งภาพผลลัพธ์กลับไปเป็น response
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        # บันทึก Error สำหรับการดีบัก
        print(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")
        # ส่ง Error กลับไป
        return Response(f"Error processing image: {e}", status=500, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
