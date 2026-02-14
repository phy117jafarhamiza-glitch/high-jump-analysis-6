import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="High Jump Smart Coach")

st.title("🏆 المدرب الذكي للوثب العالي")
st.markdown("""
**هذا النظام لا يعرض أرقاماً عشوائية، بل يقوم بـ:**
1. اكتشاف **أعلى نقطة** وصل لها اللاعب تلقائياً.
2. تحليل **لحظة الارتقاء** الحاسمة.
3. إعطاء **تقرير فني** واضح ومفهوم.
""")

# --- القائمة الجانبية ---
st.sidebar.header("إعدادات اللاعب")
athlete_height = st.sidebar.number_input("طول اللاعب (متر):", value=1.80, step=0.01)
view_side = st.sidebar.selectbox("جهة الكاميرا:", ["اليسار (Left)", "اليمين (Right)"])

# إعداد MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- دوال التحليل ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def analyze_performance(knee_angle, jump_height):
    feedback = []
    score = 0
    
    # تحليل زاوية الركبة
    if 135 <= knee_angle <= 170:
        feedback.append("✅ **زاوية الركبة (Take-off):** ممتازة! تسمح بأقصى دفع عمودي.")
        score += 1
    elif knee_angle < 135:
        feedback.append("⚠️ **زاوية الركبة:** منخفضة جداً (Deep Crouch). هذا يضيع الطاقة، حاول عدم النزول كثيراً.")
    else:
        feedback.append("⚠️ **زاوية الركبة:** مستقيمة جداً، لم تستفد من مرونة المفصل للدفع.")

    # تحليل الارتفاع (تقديري)
    if jump_height > 0.4: # 40 سم فوق الارض كمركز كتلة
        feedback.append("🚀 **الارتفاع:** جيد جداً، القوس (Arch) يبدو عالياً.")
        score += 1
    else:
        feedback.append("📉 **الارتفاع:** منخفض قليلاً، ركز على تحويل السرعة الأفقية إلى عمودية.")
        
    return feedback, score

# --- التطبيق الرئيسي ---
uploaded_file = st.file_uploader("ارفع فيديو المحاولة وسأقوم بتحليله...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # مكان عرض الفيديو والنتائج
    video_placeholder = st.empty()
    status_text = st.empty()
    
    # متغيرات لتخزين "أفضل اللقطات"
    frames_data = [] # لتخزين (الصورة، الارتفاع، الزاوية)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # تخفيف الحمل (معالجة إطار وترك إطارين) لتسريع التحليل
            if frame_count % 2 != 0:
                continue

            # تحجيم الصورة
            frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
            h, w, c = frame.shape
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # تحديد النقاط
                side = "LEFT" if view_side == "اليسار (Left)" else "RIGHT"
                hip = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y]
                knee = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].y]
                ankle = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].y]
                shoulder = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y]

                # الحسابات
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # ارتفاع مركز الكتلة (Hip Height) - معكوس لأن Y يبدأ من الأعلى
                hip_height_pixel = 1 - hip[1] 
                
                # رسم الهيكل
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # تخزين البيانات للتحليل اللاحق
                frames_data.append({
                    "frame": frame,
                    "hip_height": hip_height_pixel,
                    "knee_angle": knee_angle,
                    "frame_id": frame_count
                })
                
            # عرض الفيديو أثناء المعالجة (سريع)
            status_text.text(f"جاري تحليل الإطار رقم: {frame_count}...")
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    status_text.empty()
    video_placeholder.empty() # إخفاء الفيديو الأصلي لعرض النتائج

    # --- مرحلة "الذكاء" - تحليل البيانات المخزنة ---
    if frames_data:
        df = pd.DataFrame(frames_data)
        
        # 1. العثور على "قمة القفزة" (Max Height)
        max_height_idx = df['hip_height'].idxmax()
        peak_frame_data = df.iloc[max_height_idx]
        
        # 2. العثور على "لحظة الارتقاء" (Take-off)
        # هي اللحظة التي تسبق القمة ويكون فيها الركبة مثنية ثم تبدأ بالانفراد
        # سنبسطها بأخذ أقل ارتفاع قبل القمة
        takeoff_idx = df.iloc[:max_height_idx]['hip_height'].idxmin()
        takeoff_data = df.iloc[takeoff_idx]
        
        # --- عرض النتائج بوضوح ---
        st.success("✅ تم الانتهاء من التحليل! إليك أبرز اللقطات:")
        
        col1, col2 = st.columns(2)
        
        # عرض صورة الارتقاء
        with col1:
            st.subheader("1️⃣ لحظة الارتقاء (Take-off)")
            # رسم دائرة على الركبة وكتابة الزاوية
            img_takeoff = takeoff_data['frame'].copy()
            st.image(img_takeoff, channels="BGR", caption=f"زاوية الركبة: {int(takeoff_data['knee_angle'])} درجة", use_column_width=True)
            
        # عرض صورة القمة
        with col2:
            st.subheader("2️⃣ قمة القفزة (Peak Height)")
            img_peak = peak_frame_data['frame'].copy()
            st.image(img_peak, channels="BGR", caption="أقصى ارتفاع وصل له الورك", use_column_width=True)

        # --- تقرير المدرب (النص المفهوم) ---
        st.markdown("---")
        st.header("📝 تقرير المدرب الآلي")
        
        feedback_list, score = analyze_performance(takeoff_data['knee_angle'], peak_frame_data['hip_height'])
        
        for item in feedback_list:
            st.markdown(item)
            
        if score == 2:
            st.balloons()
            st.success("🎉 أداء ممتاز! المحاولة مثالية من الناحية الميكانيكية.")
        elif score == 1:
            st.warning("⚠️ أداء جيد، لكن هناك مجال للتحسين في النقاط المذكورة أعلاه.")
        else:
            st.error("🛑 تحتاج إلى مراجعة شاملة لتقنية القفز، انتبه للملاحظات.")

    else:
        st.error("لم يتم اكتشاف جسم اللاعب بوضوح. تأكد من الإضاءة وظهور اللاعب بالكامل.")
