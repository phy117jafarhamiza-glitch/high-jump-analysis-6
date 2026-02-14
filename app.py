import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd
import time

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="High Jump Biomechanics Lab")

st.title("🔬 مختبر التحليل الميكانيكي الحيوي الشامل")
st.markdown("""
هذا التطبيق يقوم بحساب:
1. **الزوايا:** (الركبة، الورك، قوس الظهر).
2. **الكينماتيكا:** (السرعة العمودية، السرعة الأفقية، ارتفاع الطيران).
3. **مركز الكتلة (CoM):** رسم مسار الحركة.
""")

# --- القائمة الجانبية للإعدادات ---
st.sidebar.header("⚙️ إعدادات المعايرة")
athlete_height = st.sidebar.number_input("طول اللاعب (بالمتر) - للمعايرة:", min_value=1.0, max_value=2.5, value=1.80, step=0.01)
fps_input = st.sidebar.number_input("معدل إطارات الفيديو (FPS) - تقريبي:", min_value=15, max_value=240, value=30)
view_side = st.sidebar.selectbox("جهة التصوير:", ["اليسار (Left)", "اليمين (Right)"])

# إعداد MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# دوال مساعدة
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_center_of_mass(landmarks):
    # تقريب مركز الكتلة باستخدام منتصف الوركين (نقطة مبسطة للوثب العالي)
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    center_x = (left_hip[0] + right_hip[0]) / 2
    center_y = (left_hip[1] + right_hip[1]) / 2
    return [center_x, center_y]

# --- التطبيق الرئيسي ---
uploaded_file = st.file_uploader("ارفع فيديو المحاولة (يفضل تصوير جانبي ثابت)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # واجهة العرض
    col1, col2 = st.columns([3, 2])
    with col1:
        stframe = st.empty()
    with col2:
        st.subheader("📊 القياسات الحية")
        metric_knee = st.empty()
        metric_arch = st.empty()
        metric_vel_y = st.empty()
        metric_height = st.empty()
        
    # متغيرات لتخزين البيانات للرسم البياني
    data_log = []
    prev_com_y = None
    prev_time = 0
    trajectory_points = []
    
    # المعايرة (تقديرية: نفترض أن طول الجسم في الفيديو يغطي نسبة معينة)
    # ملاحظة: المعايرة الدقيقة تتطلب معرفة طول جسم اللاعب بالبكسل في كل إطار
    # سنستخدم "Scale" بسيط يعتمد على المسافة بين الكتف والكاحل لتقريب المتر
    pixel_to_meter_scale = 0.0 # سيتم حسابه داخل الحلقة

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # تحجيم الصورة للحفاظ على الأداء
            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            h, w, c = frame.shape
            
            # معالجة Mediapipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 1. استخراج النقاط
                try:
                    # تحديد النقاط بناءً على الجهة
                    side_prefix = "LEFT" if view_side == "اليسار (Left)" else "RIGHT"
                    
                    # دالة مساعدة لجلب الإحداثيات
                    def get_lm(name):
                        lm = landmarks[getattr(mp_pose.PoseLandmark, f"{side_prefix}_{name}").value]
                        return [lm.x, lm.y]
                    
                    shoulder = get_lm("SHOULDER")
                    hip = get_lm("HIP")
                    knee = get_lm("KNEE")
                    ankle = get_lm("ANKLE")
                    
                    # 2. حساب عامل المعايرة (Scale Factor)
                    # نحسب طول اللاعب الظاهري بالبكسل (من الكتف للكاحل) لتقريب التحويل
                    pixel_height = np.linalg.norm(np.array(shoulder) - np.array(ankle)) # مسافة نسبية (0-1)
                    if pixel_height > 0.1: # لتجنب الأخطاء إذا كان الجسم بعيداً
                         # نفترض أن المسافة من الكتف للكاحل تمثل حوالي 80% من طول اللاعب الكلي
                        estimated_body_pixels = pixel_height / 0.8
                        pixel_to_meter_scale = athlete_height / estimated_body_pixels # متر لكل وحدة نسبية

                    # 3. حساب الزوايا
                    knee_angle = calculate_angle(hip, knee, ankle)
                    hip_angle = calculate_angle(shoulder, hip, knee) # زاوية القوس
                    
                    # 4. حساب مركز الكتلة (CoM) والسرعة
                    com = get_center_of_mass(landmarks) # [x, y] نسبي
                    
                    # تحويل CoM إلى بكسل للرسم
                    cx, cy = int(com[0] * w), int(com[1] * h)
                    trajectory_points.append((cx, cy))
                    
                    # حساب السرعة العمودية (Vertical Velocity)
                    current_time = time.time()
                    velocity_y = 0.0
                    jump_height = 0.0
                    
                    if prev_com_y is not None and pixel_to_meter_scale > 0:
                        # الفرق في المسافة (y inverted because 0 is top)
                        delta_y = (prev_com_y - com[1]) * pixel_to_meter_scale # بالمتر
                        delta_t = 1.0 / fps_input # الزمن بالثواني بناء على الـ FPS
                        
                        velocity_y = delta_y / delta_t # م/ث
                        
                        # حساب ارتفاع القفزة التقريبي (من الأرض)
                        # نفترض أن الكاحل هو الأرض تقريباً
                        jump_height = (ankle[1] - com[1]) * pixel_to_meter_scale
                    
                    prev_com_y = com[1]

                    # --- الرسم على الفيديو ---
                    # رسم المسار (Trajectory)
                    for i in range(1, len(trajectory_points)):
                        cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 255, 255), 2)
                    
                    # رسم نقطة مركز الكتلة
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                    
                    # رسم الزوايا
                    knee_pos = tuple(np.multiply(knee, [w, h]).astype(int))
                    cv2.putText(frame, f"{int(knee_angle)} deg", knee_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # تحديث البيانات الحية
                    metric_knee.metric("زاوية الركبة", f"{int(knee_angle)}°")
                    metric_arch.metric("زاوية القوس (Hip)", f"{int(hip_angle)}°")
                    metric_vel_y.metric("السرعة العمودية", f"{velocity_y:.2f} m/s")
                    metric_height.metric("ارتفاع مركز الكتلة", f"{jump_height:.2f} m")
                    
                    # تخزين البيانات للتحليل النهائي
                    data_log.append({
                        "Frame": len(data_log),
                        "Knee Angle": knee_angle,
                        "Hip Angle": hip_angle,
                        "Vertical Velocity (m/s)": velocity_y,
                        "CoM Height (m)": jump_height
                    })
                    
                except Exception as e:
                    pass

            # عرض الفيديو
            stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    # --- عرض الرسوم البيانية بعد انتهاء الفيديو ---
    st.markdown("---")
    st.subheader("📈 تحليل الأداء البياني (Performance Analytics)")
    
    if data_log:
        df = pd.DataFrame(data_log)
        
        # رسم 1: السرعة والارتفاع
        st.write("### تغير السرعة العمودية والارتفاع")
        st.line_chart(df[["Vertical Velocity (m/s)", "CoM Height (m)"]])
        
        # رسم 2: الزوايا
        st.write("### تغير زوايا المفاصل (Kinematics)")
        st.line_chart(df[["Knee Angle", "Hip Angle"]])
        
        # جدول البيانات الخام (للمدربين)
        with st.expander("عرض البيانات الخام (Excel)"):
            st.dataframe(df)
            
            # زر التحميل (CSV)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="تحميل تقرير التحليل (CSV)",
                data=csv,
                file_name='jump_analysis.csv',
                mime='text/csv',
            )
