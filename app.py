import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# إعداد الصفحة
st.title("تطبيق تحليل القفز العالي (CNN)")
st.write("قم برفع فيديو لمحاولة القفز لتحليل زاوية الركبة واكتشاف الأخطاء.")

# إعداد MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# دالة حساب الزاوية
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# رفع الفيديو
uploaded_file = st.file_uploader("اختر ملف فيديو...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    kpi_text = st.empty()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # تحويل الألوان
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # محاولة استخراج النقاط
            try:
                landmarks = results.pose_landmarks.landmark
                
                # استخراج الإحداثيات (الورك، الركبة، الكاحل - يسار)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # حساب الزاوية
                angle = calculate_angle(hip, knee, ankle)
                
                # عرض الزاوية على الشاشة
                cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                diagnosis = ""
                if angle < 140:
                    diagnosis = "⚠️ انثناء زائد (Deep Knee Bend)"
                else:
                    diagnosis = "✅ وضع جيد"

                kpi_text.markdown(f"### زاوية الركبة: **{int(angle)}** درجة\n#### التشخيص: {diagnosis}")
                
            except:
                pass
            
            # رسم الهيكل
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            stframe.image(image, channels="BGR", use_column_width=True)

    cap.release()
