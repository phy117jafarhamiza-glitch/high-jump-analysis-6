import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="High Jump Pro Report")

# العنوان
st.markdown("""
<style>
.big-font { font-size:30px !important; font-weight: bold; color: #1f77b4; }
.feedback-box { padding: 15px; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📋 تقرير الأداء الفني للوثب العالي</p>', unsafe_allow_html=True)
st.write("نظام تحليل الاداء للمدربين واللاعبين: أرقام دقيقة + مقارنة معيارية + توجيهات.")

# --- الشريط الجانبي ---
st.sidebar.header("1️⃣ بيانات اللاعب")
athlete_height = st.sidebar.number_input("طول اللاعب (متر):", value=1.80, step=0.01)
view_side = st.sidebar.selectbox("جهة التصوير:", ["اليسار (Left)", "اليمين (Right)"])
st.sidebar.warning("⚠️ تأكد أن التصوير جانبي وثابت للحصول على تقرير دقيق.")

# --- دوال الحساب ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_feedback(metric_name, value, target_min, target_max):
    if target_min <= value <= target_max:
        return "✅ ممتاز (ضمن النطاق المثالي)"
    elif value < target_min:
        return "⚠️ منخفض (تحتاج لزيادة الرقم)"
    else:
        return "⚠️ مرتفع (تحتاج لتقليل الرقم)"

# --- المحرك الرئيسي ---
uploaded_file = st.file_uploader("2️⃣ ارفع فيديو المحاولة هنا", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # واجهة العرض أثناء التحليل
    col1, col2 = st.columns([2, 1])
    with col1:
        st_video = st.empty()
    with col2:
        st.info("جاري تحليل المحاولة... يرجى الانتظار")
        st_progress = st.progress(0)
    
    # متغيرات لتخزين "أقصى" و "أدنى" قيم خلال المحاولة
    min_knee_angle = 180 # نبحث عن أقل زاوية (لحظة التحميل)
    max_hip_height = 0   # نبحث عن أعلى ارتفاع
    max_velocity = 0     # أقصى سرعة عمودية
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # لحساب السرعة
    prev_hip_y = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            if total_frames > 0: st_progress.progress(current_frame / total_frames)

            # معالجة الصورة
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                side = "LEFT" if view_side == "اليسار (Left)" else "RIGHT"
                
                # النقاط
                hip = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y]
                knee = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].y]
                ankle = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].y]
                shoulder = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x, landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y]

                # 1. الحسابات اللحظية
                # زاوية الركبة
                angle = calculate_angle(hip, knee, ankle)
                if angle < min_knee_angle: min_knee_angle = angle # تسجيل أعمق نقطة نزول
                
                # الارتفاع والسرعة (بالمتر التقديري)
                # عامل التحويل: طول الجذع (كتف-ورك) يمثل تقريباً 30% من طول اللاعب
                torso_len_pixel = np.linalg.norm(np.array(shoulder) - np.array(hip))
                if torso_len_pixel > 0:
                    pixel_to_meter = (athlete_height * 0.3) / torso_len_pixel
                    
                    # ارتفاع الورك عن "أسفل الصورة" (تقديري للأرض)
                    current_height_m = (1 - hip[1]) * pixel_to_meter * 3.3 # تقريب
                    if current_height_m > max_hip_height: max_hip_height = current_height_m
                    
                    # السرعة العمودية
                    if prev_hip_y is not None:
                        dist_pixel = prev_hip_y - hip[1] # المسافة المقطوعة
                        dist_m = dist_pixel * pixel_to_meter * 3.3
                        vel = dist_m * fps # السرعة م/ث
                        if vel > max_velocity and vel < 10: # تجاهل الأخطاء (أقل من 10م/ث)
                            max_velocity = vel
                    
                    prev_hip_y = hip[1]
                
                # رسم
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            st_video.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    st_progress.empty() # إخفاء شريط التحميل

    # --- 3️⃣ عرض التقرير النهائي (هنا الزبدة) ---
    st.markdown("---")
    st.subheader("📊 نتائج التحليل وتوصيات المدرب")

    # تقسيم التقرير إلى 3 أعمدة
    col1, col2, col3 = st.columns(3)

    # المؤشر 1: زاوية التحميل (الركبة)
    with col1:
        st.markdown("### 🦵 زاوية الركبة")
        val = int(min_knee_angle)
        status = get_feedback("Knee", val, 135, 145)
        color = "green" if "ممتاز" in status else "red"
        
        st.metric(label="أقل زاوية تم تسجيلها", value=f"{val}°", delta=status, delta_color="inverse" if "مرتفع" in status else "normal")
        st.markdown(f"""
        **المعيار المثالي:** 135° - 145°
        
        **التغذية الراجعة:**
        {'انثناء الركبة ممتاز ويسمح بتخزين طاقة قوية.' if val >= 135 and val <= 145 else 'الركبة انثنت بشكل زائد (جلوس)، هذا يقتل السرعة.' if val < 135 else 'النزول غير كافي، لم تستفد من قوة الفخذين.'}
        """)

    # المؤشر 2: السرعة العمودية
    with col2:
        st.markdown("### 🚀 قوة الانفجار")
        val = round(max_velocity, 2)
        status = "جيد" if val > 3.0 else "يحتاج تحسين"
        
        st.metric(label="أقصى سرعة عمودية", value=f"{val} m/s")
        st.markdown(f"""
        **المعيار:** > 3.5 m/s (للمحترفين)
        
        **التغذية الراجعة:**
        سرعة الانطلاق هي مفتاح الارتفاع.
        {'دفعك للأرض قوي جداً!' if val > 4.0 else 'دفعك جيد، لكن حاول تسريع الخطوة الأخيرة.' if val > 3.0 else 'الدفع بطيء، تحتاج لتمارين قوة انفجارية (Plyometrics).'}
        """)

    # المؤشر 3: ارتفاع الطيران
    with col3:
        st.markdown("### 📈 ارتفاع الورك")
        val = round(max_hip_height, 2)
        
        st.metric(label="أقصى ارتفاع للورك", value=f"{val} m")
        st.markdown(f"""
        **المقارنة بطولك:** {round(val - (athlete_height/2), 2)}+ متر
        
        **التغذية الراجعة:**
        هذا الرقم يعكس كفاءة القوس (Arch) فوق العارضة. تأكد من رفع الورك لأعلى نقطة ممكنة لحظة العبور.
