import cv2
from ultralytics import YOLO

# --- 1. โหลดโมเดลที่เราฝึกมา ---
# ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกับไฟล์นี้
model = YOLO('steel.pt')

# --- 2. เปิดกล้องและตั้งค่า ROI อัตโนมัติ ---
# ใช้ index 1 สำหรับ Camo หรือ 0 สำหรับเว็บแคมทั่วไป
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องได้")
    exit()

# อ่านเฟรมแรกเพื่อนำมาหาขนาด
ret, frame = cap.read()
if not ret:
    print("Error: ไม่สามารถอ่านเฟรมแรกจากกล้องได้")
    cap.release()
    exit()

# หาขนาดของเฟรม (กว้าง x สูง)
frame_height, frame_width, _ = frame.shape
print(f"ตรวจพบขนาดกล้อง: {frame_width} x {frame_height}")

# สร้างกรอบ ROI โดยเว้นขอบล่างและขวาให้มากขึ้นเพื่อหลบลายน้ำ
border_left = 5
border_top = 5
border_right = 100 # ขอบด้านขวา (ขยับเข้ามากหน่อยเพื่อหลบโลโก้)
border_bottom = 100 # ขอบด้านล่าง (ขยับเข้ามากหน่อยเพื่อหลบโลโก้)

roi_x = border_left
roi_y = border_top
roi_w = frame_width - border_left - border_right
roi_h = frame_height - border_top - border_bottom

# --- 3. ลูปการทำงานแบบเรียลไทม์ ---
while True:
    # อ่านเฟรมจากกล้องS
    ret, frame = cap.read()
    if not ret:
        break

    # --- 4. ประมวลผลเฉพาะในกรอบ (ROI) ---
    # ตัดภาพให้เหลือเฉพาะในกรอบก่อนส่งให้โมเดล
    roi_frame = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

    # ส่งเฉพาะภาพในกรอบให้โมเดลประมวลผล
    results = model(roi_frame, conf=0.7)

    # --- 5. แสดงผลลัพธ์ ---
    # .plot() จะวาดกรอบลงบน roi_frame ที่เราส่งเข้าไป
    annotated_roi = results[0].plot(font_size=0.1, line_width=1) 
    
    # นำภาพในกรอบที่วาดผลลัพธ์แล้ว ไปแปะกลับคืนที่ตำแหน่งเดิมบนภาพเต็ม
    frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = annotated_roi
    
    # วาดกรอบสีเหลืองทับลงบนภาพเต็มเพื่อแสดงขอบเขต
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 3)
    
    # นับจำนวนเหล็กที่เจอ (จากผลลัพธ์ในกรอบ)
    steel_bar_count = len(results[0].boxes)
    text = f"Total Steel Bars: {steel_bar_count}"

    print(text) # <-- เพิ่มบรรทัดนี้เพื่อพิมพ์จำนวนใน Terminal
    
    # ใส่ข้อความจำนวนนับลงบนภาพ
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 255, 0), 3)

    # แสดงภาพผลลัพธ์
    cv2.imshow("YOLOv8 Steel Bar Counter", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. คืนทรัพยากร ---
cap.release()
cv2.destroyAllWindows()