import streamlit as st
import requests
import base64
import io
import time
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# ---- External / local modules ----
from firebase_auth import (
    init_firebase_auth,
    check_authentication_status,
    set_authentication_status,
)

# Make user_history optional: don't crash if the module isn't present
try:
    from user_history import (
        display_user_history,
        display_report_details,
        create_report_summary_card,
    )
except Exception:
    def display_user_history(*args, **kwargs):
        pass
    def display_report_details(*args, **kwargs):
        pass
    def create_report_summary_card(*args, **kwargs):
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .medical-report {
        background-color: #f8f9fa;
        color: #212529;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .malignancy-malignant { color: #dc3545; font-weight: bold; }
    .malignancy-benign { color: #28a745; font-weight: bold; }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Config
BACKEND_URL = "http://localhost:5000"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers

def _sanitize_filename(value: str, fallback: str = "report") -> str:
    if not value:
        return fallback
    bad = '<>:"/\\|?*'
    for ch in bad:
        value = value.replace(ch, "_")
    return value.strip() or fallback

def _get(d: dict, path: str, default=None):
    """Safe dict getter for dotted paths: _get(pred, 'a.b.c')"""
    try:
        cur = d
        for part in path.split("."):
            cur = cur.get(part, {})
        return cur if cur != {} else default
    except Exception:
        return default

def _decode_data_url_or_b64(b64_str: str) -> bytes | None:
    """Accepts plain base64 or data URLs (data:image/png;base64,XXXX)."""
    if not b64_str:
        return None
    try:
        if b64_str.startswith("data:"):
            # Split off header "data:mime/type;base64,"
            header, _, data = b64_str.partition(",")
            return base64.b64decode(data)
        return base64.b64decode(b64_str)
    except Exception:
        return None

@st.cache_resource
def get_firebase_auth():
    """Initialize and cache Firebase authentication resource."""
    return init_firebase_auth()

def show_login_page():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Please log in or sign up to continue</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    firebase_auth = get_firebase_auth()

    with tab1:
        st.markdown("### Login to Your Account")
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login", type="primary")
        if login_submitted:
            if email and password:
                with st.spinner("Logging in..."):
                    result = firebase_auth.sign_in(email, password)
                if result.get("success"):
                    set_authentication_status(True, result.get("user"))
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error(f"❌ Login failed: {result.get('error', 'Unknown error')}")
            else:
                st.error("Please enter both email and password.")

    with tab2:
        st.markdown("### Create New Account")
        with st.form("signup_form"):
            name = st.text_input("Full Name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            signup_submitted = st.form_submit_button("Sign Up", type="primary")
        if signup_submitted:
            if not all([name, email, password, confirm_password]):
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                with st.spinner("Creating account..."):
                    result = firebase_auth.sign_up(email, password, name)
                if result.get("success"):
                    set_authentication_status(True, result.get("user"))
                    st.success("✅ Account created successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Sign up failed: {result.get('error', 'Unknown error')}")

def show_logout_button():
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout"):
            set_authentication_status(False, None)
            st.rerun()

def check_backend_health() -> tuple[bool, dict | None]:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            try:
                return True, response.json()
            except Exception:
                return True, {}
        return False, None
    except requests.exceptions.RequestException:
        return False, None

def upload_image_to_backend(uploaded_file) -> dict | None:
    """
    Upload image to backend for prediction using a proper multipart tuple.
    """
    try:
        uploaded_file.seek(0)
        files = {
            'image': (
                getattr(uploaded_file, "name", "upload.png"),
                uploaded_file.read(),
                getattr(uploaded_file, "type", "application/octet-stream"),
            )
        }
        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=60)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                st.error("API returned non-JSON body.")
                return None
        else:
            # include small excerpt of the response for debugging
            text_snippet = (response.text or "")[:500]
            st.error(f"API Error: {response.status_code} – {text_snippet}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def generate_pdf_report(prediction: dict, medical_summary: str) -> bytes:
    """
    Generate a PDF report with the medical analysis results.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )

    story = []
    story.append(Paragraph("BRAIN TUMOR DETECTION REPORT", title_style))
    story.append(Spacer(1, 20))

    tumor_type = (prediction or {}).get('tumor_type', 'unknown').replace('_', ' ').title()
    malignancy = (prediction or {}).get('malignancy', 'unknown').title()
    confidence = (prediction or {}).get('confidence', 0.0)
    try:
        conf_str = f"{float(confidence):.1%}"
    except Exception:
        conf_str = "N/A"

    data = [
        ['Finding', 'Value'],
        ['Tumor Type', tumor_type],
        ['Malignancy', malignancy],
        ['Confidence', conf_str],
    ]

    table = Table(data, colWidths=[2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("MEDICAL ANALYSIS REPORT", heading_style))

    # Render the summary line-by-line; treat some as headings if they start with known prefixes
    summary_lines = (medical_summary or "").split('\n')
    heading_prefixes = (
        'DIAGNOSIS:', 'CONFIDENCE:', 'CLINICAL FINDINGS:',
        'SEVERITY ASSESSMENT:', 'ANATOMICAL LOCATION:',
        'CLINICAL IMPLICATIONS:', 'RECOMMENDATIONS:', 'NOTE:'
    )
    for line in summary_lines:
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        if any(line.strip().startswith(prefix) for prefix in heading_prefixes):
            story.append(Paragraph(line, heading_style))
        else:
            story.append(Paragraph(line, normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def display_prediction_results(result: dict):
    """
    Display prediction results (no saving here to avoid duplication).
    """
    prediction = result.get('prediction', {})
    medical_summary = result.get('medical_summary', '')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Prediction Results")

        tumor_type_raw = prediction.get('tumor_type', 'unknown')
        tumor_type = tumor_type_raw.replace('_', ' ').title()
        malignancy_raw = prediction.get('malignancy', 'unknown')
        malignancy = malignancy_raw.title()
        confidence = float(prediction.get('confidence', 0.0))

        # Colors for UI
        if confidence >= 0.85:
            confidence_class = "confidence-high"
        elif confidence >= 0.70:
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"

        malignancy_class = f"malignancy-{malignancy_raw}"

        st.markdown(f"""
        <div class="prediction-box">
            <h4>🏥 Detected Tumor Type: <strong>{tumor_type}</strong></h4>
            <p class="{malignancy_class}">🔬 Malignancy: <strong>{malignancy}</strong></p>
            <p class="{confidence_class}">🎯 Confidence: <strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Probability Distribution
        st.markdown("### 📈 Probability Distribution")
        probs = prediction.get('probabilities') or {}
        if probs:
            try:
                import plotly.graph_objects as go
                bars = go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=[
                        '#1f77b4' if k == tumor_type_raw else '#ff7f0e'
                        for k in probs.keys()
                    ],
                    text=[f"{float(v):.1%}" for v in probs.values()],
                    textposition='auto',
                )
                fig = go.Figure(data=[bars])
                fig.update_layout(
                    title="Tumor Type Probabilities",
                    xaxis_title="Tumor Type",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render probability chart: {e}")
        else:
            st.info("No probability distribution provided by backend.")

    with col2:
        st.markdown("### 🔍 Grad-CAM Visualization")
        gradcam_b64 = _get(result, "visualization.gradcam_image")
        img_bytes = _decode_data_url_or_b64(gradcam_b64) if gradcam_b64 else None
        if img_bytes:
            try:
                gradcam_image = Image.open(io.BytesIO(img_bytes))
                st.image(gradcam_image, caption="Grad-CAM Analysis", use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM image could not be displayed: {e}")
        else:
            st.warning("Grad-CAM visualization not available.")

def show_analysis_tab(firebase_auth, user_info):
    st.markdown("### 📤 Upload MRI Image")
    st.markdown("Please upload a brain MRI image for tumor detection analysis.")

    uploaded_file = st.file_uploader(
        "Choose an MRI image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )

    if uploaded_file is not None and st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Analyzing image... This may take a few moments."):
            # Call backend
            result = upload_image_to_backend(uploaded_file)
        if result and result.get('success'):
            st.success("✅ Analysis completed successfully!")
            # Persist for Reports tab
            st.session_state.last_result = result
            # Save report (only here; not inside display function)
            try:
                report_data = {
                    'prediction': result.get('prediction', {}),
                    'medical_summary': result.get('medical_summary', ''),
                    'visualization': result.get('visualization', {}),
                    'audio': result.get('audio', {}),
                    'metadata': result.get('metadata', {}),
                }
                firebase_auth.save_report(user_info['uid'], report_data)
            except Exception as e:
                st.warning(f"Could not save report: {e}")
            # Reset cached PDF so a fresh one is generated
            st.session_state.pop('pdf_data', None)
        else:
            st.error("❌ Analysis failed. Please try again.")

    # Render core results if available
    result = st.session_state.get('last_result')
    if result and result.get('success'):
        display_prediction_results(result)

def show_reports_feedback_tab(firebase_auth, user_info):
    result = st.session_state.get('last_result')
    if not result or not result.get('success'):
        st.info("Run an analysis on the Analysis tab to see reports and downloads here.")
        return

    prediction = result.get('prediction', {})
    medical_summary = result.get('medical_summary', '')

    st.markdown("### 📋 Medical Analysis Report")
    st.markdown(f'<div class="medical-report">{medical_summary}</div>', unsafe_allow_html=True)

    st.markdown("### 📄 Downloads")
    # Cache PDF in session
    if 'pdf_data' not in st.session_state or st.session_state.get('pdf_data') is None:
        try:
            st.session_state.pdf_data = generate_pdf_report(prediction, medical_summary)
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
            st.session_state.pdf_data = None

    tumor_type_for_file = _sanitize_filename(prediction.get('tumor_type', 'unknown'))
    malignancy_for_file = _sanitize_filename(prediction.get('malignancy', 'unknown'))

    if st.session_state.get('pdf_data'):
        st.download_button(
            label="📥 Download Medical Report (PDF)",
            data=st.session_state.pdf_data,
            file_name=f"brain_tumor_report_{tumor_type_for_file}_{malignancy_for_file}.pdf",
            mime="application/pdf",
            key="pdf_download_main"
        )

    # Audio
    audio_info = result.get('audio', {})
    if audio_info.get('available') and audio_info.get('audio_data'):
        audio_bytes = _decode_data_url_or_b64(audio_info['audio_data'])
        if audio_bytes:
            st.markdown("### 🔊 Audio Summary")
            st.audio(audio_bytes, format='audio/mp3')
            st.download_button(
                label="📥 Download Audio Summary",
                data=audio_bytes,
                file_name=f"medical_summary_{tumor_type_for_file}.mp3",
                mime="audio/mp3",
                key="audio_download_main"
            )

    st.markdown("### 💬 Feedback")
    with st.form("feedback_form"):
        rating = st.slider("Overall rating", 1, 5, 5)
        feedback_text = st.text_area("Your feedback", height=150, placeholder="What worked well? What can be improved?")
        submitted = st.form_submit_button("Submit Feedback", type="primary")
    if submitted:
        if not feedback_text.strip():
            st.error("Please enter feedback before submitting.")
        else:
            with st.spinner("Submitting feedback..."):
                ok = firebase_auth.save_feedback(
                    user_info['uid'],
                    feedback_text.strip(),
                    rating,
                    metadata={"app_version": "1.0", "frontend": "streamlit"}
                )
            if ok:
                st.success("Thank you! Your feedback was submitted.")
            else:
                st.error("Could not submit feedback at this time.")

def show_system_status_page(backend_healthy: bool, health_info: dict | None):
    st.markdown("### 🔧 System Status")
    if backend_healthy:
        st.success("✅ Backend API Connected")
        if health_info is not None:
            st.info(f"Model: {'✅ Loaded' if health_info.get('model_loaded') else '❌ Not Loaded'}")
            st.info(f"TTS: {'✅ Available' if health_info.get('tts_available') else '❌ Not Available'}")
    else:
        st.error("❌ Backend API Not Available")
        st.info("To start the backend, run: `python backend.py`")

    # Firebase status
    try:
        firebase_auth = get_firebase_auth()
        if getattr(firebase_auth, "db", None):
            st.success("✅ Firebase Connected")
        else:
            st.warning("⚠️ Firebase Not Available (using session-based auth)")
    except Exception:
        st.warning("⚠️ Firebase Not Available (using session-based auth)")

def main():
    # Ensure session keys exist
    st.session_state.setdefault('last_result', None)
    st.session_state.setdefault('pdf_data', None)

    # Auth
    auth_status = check_authentication_status()
    if not auth_status.get('authenticated'):
        show_login_page()
        return

    user_info = auth_status.get('user_info') or {}
    firebase_auth = get_firebase_auth()

    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.2rem; color: #666;">Welcome, {user_info.get("name", "User")}!</p>', unsafe_allow_html=True)

    # Backend health
    backend_healthy, health_info = check_backend_health()
    if not backend_healthy:
        st.error("⚠️ Backend API is not available. Please ensure the Flask backend is running on localhost:5000")
        st.info("To start the backend, run: `python backend.py`")
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 User: {user_info.get('name', 'User')}")
        if user_info.get('email'):
            st.markdown(f"📧 {user_info['email']}")
        st.markdown("---")
    show_logout_button()

    # Tabs
    page1, page2 = st.tabs(["🔍 Analysis", "📄 Reports & Feedback"])
    with page1:
        show_analysis_tab(firebase_auth, user_info)
    with page2:
        show_reports_feedback_tab(firebase_auth, user_info)

if __name__ == "__main__":
    main()