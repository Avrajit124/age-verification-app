import streamlit as st
import cv2
import numpy as np
import time
import threading
import base64
import os

from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Age Verification",
    page_icon="🔐",
    layout="centered"
)


# ============================================================
# BACKGROUND
# ============================================================

BACKGROUND_PATH = "background.png"

if os.path.exists(BACKGROUND_PATH):

    with open(BACKGROUND_PATH, "rb") as f:
        background_data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        [data-testid="stAppViewContainer"] {{
            background-image: url(
                "data:image/png;base64,{background_data}"
            );

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .main {{
            background: transparent;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
        opacity: 0.85;
    }

    .info-card {
        background: rgba(255,255,255,0.94);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: #111111;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.12);
    }

    .info-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .info-line {
        font-size: 16px;
        margin: 7px 0;
    }

    .camera-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .status-box {
        text-align: center;
        padding: 15px;
        margin-top: 12px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 800;
        background: #000000;
        color: #ffffff;
    }

    .warning {
        text-align: center;
        margin-top: 20px;
        font-size: 13px;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔐 Age Verification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered real-time facial age estimation</div>',
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION CARD
# ============================================================

st.markdown(
    """
    <div class="info-card">

        <div class="info-title">How does it work?</div>

        <div class="info-line">
            1️⃣ Position your face inside the guide box.
        </div>

        <div class="info-line">
            2️⃣ Keep your face steady while the 10-second verification runs.
        </div>

        <div class="info-line">
            3️⃣ At the end, one face image is captured and analyzed by AI.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GLOBAL RUNTIME STATE
# ============================================================

class VerificationState:

    def __init__(self):

        self.lock = threading.Lock()

        # Face information
        self.face_box = None
        self.face_valid = False
        self.last_face_time = 0.0

        # Stability
        self.stable_start = None

        # Verification timer
        self.verification_started = False
        self.verification_start_time = None

        # Capture
        self.capture_requested = False
        self.captured_face = None
        self.capture_done = False

        # Prediction
        self.prediction_started = False
        self.prediction_done = False
        self.prediction_result = None

        # Camera frame
        self.last_frame = None

        # Detection frame counter
        self.frame_counter = 0

        # Final state
        self.finished = False

        # Error
        self.error = None

    def reset(self):

        with self.lock:

            self.face_box = None
            self.face_valid = False
            self.last_face_time = 0.0

            self.stable_start = None

            self.verification_started = False
            self.verification_start_time = None

            self.capture_requested = False
            self.captured_face = None
            self.capture_done = False

            self.prediction_started = False
            self.prediction_done = False
            self.prediction_result = None

            self.last_frame = None

            self.frame_counter = 0

            self.finished = False

            self.error = None


# One state object for the running Streamlit process.
if "runtime_state" not in st.session_state:

    st.session_state.runtime_state = VerificationState()

state = st.session_state.runtime_state


# ============================================================
# HAAR CASCADE
# ============================================================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)


# ============================================================
# MODEL
# ============================================================

MODEL_REPO = "onnx-community/age-gender-prediction-ONNX"
MODEL_FILENAME = "onnx/model.onnx"


@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    return session


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(frame):

    """
    Lightweight face detection.

    Detection is performed on a smaller image to reduce CPU usage.
    """

    if frame is None:
        return None

    height, width = frame.shape[:2]

    # Smaller detection image = much faster processing
    scale = 0.35

    small = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(55, 55)
    )

    if len(faces) == 0:
        return None

    # Choose largest face
    x, y, w, h = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    # Convert coordinates back to original frame
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    return x, y, w, h


# ============================================================
# FACE VALIDATION
# ============================================================

def is_good_face(frame, box):

    if frame is None or box is None:
        return False

    h, w = frame.shape[:2]

    x, y, fw, fh = box

    if fw <= 0 or fh <= 0:
        return False

    # Face should be reasonably large
    face_ratio = (fw * fh) / float(w * h)

    if face_ratio < 0.035:
        return False

    # Face should not touch the edges
    margin_x = int(w * 0.04)
    margin_y = int(h * 0.04)

    if x < margin_x:
        return False

    if y < margin_y:
        return False

    if x + fw > w - margin_x:
        return False

    if y + fh > h - margin_y:
        return False

    return True


# ============================================================
# CAPTURE FACE CROP
# ============================================================

def crop_face(frame, box):

    if frame is None or box is None:
        return None

    x, y, w, h = box

    frame_h, frame_w = frame.shape[:2]

    # Add some margin around face
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.25)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(frame_w, x + w + margin_x)
    y2 = min(frame_h, y + h + margin_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop.copy()


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def prepare_face(face):

    """
    Prepare ONE captured face for the ONNX model.
    """

    rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    image = image.resize(
        (224, 224)
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalize to 0-1
    image = image / 255.0

    # ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (image - mean) / std

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(np.float32)


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(face):

    session = load_model()

    input_name = session.get_inputs()[0].name

    input_tensor = prepare_face(face)

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    # The model's first output value is the age estimate.
    age_value = float(
        np.asarray(outputs[0]).reshape(-1)[0]
    )

    return age_value


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    """
    IMPORTANT:

    This function must remain lightweight.

    NO ONNX MODEL HERE.

    Face detection only happens every few frames.
    """

    image = frame.to_ndarray(
        format="bgr24"
    )

    now = time.monotonic()

    with state.lock:

        state.frame_counter += 1

        frame_number = state.frame_counter

        # Keep latest frame
        state.last_frame = image.copy()

        current_box = state.face_box

        verification_started = state.verification_started

        capture_requested = state.capture_requested

    # ========================================================
    # FACE DETECTION
    # ========================================================

    # Detect only every 12 frames.
    # At ~30 FPS this means roughly 2-3 detections/sec.
    DETECTION_INTERVAL = 12

    if frame_number % DETECTION_INTERVAL == 0:

        detected_box = detect_face(image)

        valid = is_good_face(
            image,
            detected_box
        )

        with state.lock:

            if valid:

                state.face_box = detected_box
                state.face_valid = True
                state.last_face_time = now

            else:

                state.face_valid = False

    else:

        # Don't immediately declare invalid between detection frames.
        # This keeps the video smooth.
        with state.lock:

            if (
                state.last_face_time > 0
                and now - state.last_face_time < 0.7
            ):

                state.face_valid = True

            else:

                state.face_valid = False

    # ========================================================
    # GET CURRENT STATE
    # ========================================================

    with state.lock:

        box = state.face_box
        valid = state.face_valid
        started = state.verification_started
        start_time = state.verification_start_time
        capture_requested = state.capture_requested
        finished = state.finished

    # ========================================================
    # STABILITY CHECK
    # ========================================================

    if not started and not finished:

        if valid:

            with state.lock:

                if state.stable_start is None:

                    state.stable_start = now

                stable_time = now - state.stable_start

                # Face must stay stable for ~1 sec
                if stable_time >= 1.0:

                    state.verification_started = True
                    state.verification_start_time = now

        else:

            with state.lock:
                state.stable_start = None

    # ========================================================
    # 10 SECOND TIMER
    # ========================================================

    with state.lock:

        started = state.verification_started
        start_time = state.verification_start_time

    if started and start_time is not None:

        elapsed = now - start_time

        # Exactly 10 seconds
        if elapsed >= 10.0:

            with state.lock:

                if (
                    not state.capture_done
                    and not state.capture_requested
                    and state.face_valid
                    and state.last_frame is not None
                    and state.face_box is not None
                ):

                    # =================================================
                    # ONE AND ONLY ONE CAPTURE
                    # =================================================

                    captured = crop_face(
                        state.last_frame,
                        state.face_box
                    )

                    if captured is not None:

                        state.captured_face = captured
                        state.capture_requested = True
                        state.capture_done = True

    # ========================================================
    # DRAW UI
    # ========================================================

    display = image.copy()

    with state.lock:

        box = state.face_box
        valid = state.face_valid
        started = state.verification_started
        start_time = state.verification_start_time
        capture_requested = state.capture_requested
        finished = state.finished
        prediction_done = state.prediction_done

    # --------------------------------------------------------
    # GUIDE BOX
    # --------------------------------------------------------

    frame_h, frame_w = display.shape[:2]

    guide_w = int(frame_w * 0.52)
    guide_h = int(frame_h * 0.70)

    guide_x = (frame_w - guide_w) // 2
    guide_y = (frame_h - guide_h) // 2

    if finished:

        guide_color = (0, 255, 0)

    elif valid:

        guide_color = (0, 255, 0)

    else:

        guide_color = (0, 0, 255)

    cv2.rectangle(
        display,
        (guide_x, guide_y),
        (guide_x + guide_w, guide_y + guide_h),
        guide_color,
        3
    )

    # --------------------------------------------------------
    # FACE BOX
    # --------------------------------------------------------

    if box is not None:

        x, y, w, h = box

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            guide_color,
            2
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status_text = ""

    if finished:

        with state.lock:
            result = state.prediction_result

        if result == "GREATER THAN 18":

            status_text = "GREATER THAN 18"

        elif result == "LESS THAN 18":

            status_text = "LESS THAN 18"

        else:

            status_text = "PROCESSING..."

    elif capture_requested:

        status_text = "PROCESSING..."

    elif started and start_time is not None:

        elapsed = now - start_time

        remaining = max(
            0.0,
            10.0 - elapsed
        )

        status_text = (
            f"KEEP YOUR FACE IN FRAME • {remaining:.1f}s"
        )

    elif valid:

        status_text = "FACE READY ✓"

    else:

        status_text = "POSITION YOUR FACE INSIDE THE BOX"

    # --------------------------------------------------------
    # BLACK STATUS BOX
    # --------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.75
    thickness = 2

    text_size = cv2.getTextSize(
        status_text,
        font,
        font_scale,
        thickness
    )[0]

    text_x = int(
        (frame_w - text_size[0]) / 2
    )

    text_y = frame_h - 35

    # Black rectangle
    box_padding_x = 20
    box_padding_y = 12

    rect_x1 = max(
        10,
        text_x - box_padding_x
    )

    rect_y1 = max(
        10,
        text_y - text_size[1] - box_padding_y
    )

    rect_x2 = min(
        frame_w - 10,
        text_x + text_size[0] + box_padding_x
    )

    rect_y2 = min(
        frame_h - 10,
        text_y + box_padding_y
    )

    cv2.rectangle(
        display,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        status_text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # ========================================================
    # RETURN FRAME
    # ========================================================

    return frame.from_ndarray(
        display,
        format="bgr24"
    )


# ============================================================
# CAMERA
# ============================================================

st.markdown(
    '<div class="camera-title">📷 Camera</div>',
    unsafe_allow_html=True
)


ctx = webrtc_streamer(
    key="age-verification-camera",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302"
                ]
            }
        ]
    },

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True,

    video_frame_callback=video_frame_callback
)


# ============================================================
# PREDICTION WORKER
# ============================================================

@st.fragment(run_every=0.25)
def prediction_worker():

    with state.lock:

        capture_requested = state.capture_requested
        prediction_started = state.prediction_started
        prediction_done = state.prediction_done
        captured_face = state.captured_face
        finished = state.finished
        error = state.error

    # ========================================================
    # RUN ONLY ONCE AFTER 10 SECONDS
    # ========================================================

    if (
        capture_requested
        and not prediction_started
        and not prediction_done
        and captured_face is not None
    ):

        with state.lock:

            state.prediction_started = True

        try:

            # ================================================
            # ONLY NOW RUN THE AI MODEL
            # ================================================

            age = predict_age(
                captured_face
            )

            # >= 18 means greater than 18 category
            if age >= 18:

                result = "GREATER THAN 18"

            else:

                result = "LESS THAN 18"

            with state.lock:

                state.prediction_result = result
                state.prediction_done = True
                state.finished = True

        except Exception as e:

            with state.lock:

                state.error = str(e)
                state.prediction_done = True
                state.finished = True

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    with state.lock:

        result = state.prediction_result
        finished = state.finished
        error = state.error

    if error:

        st.error(
            "Unable to analyze the captured face."
        )

    elif finished and result:

        if result == "GREATER THAN 18":

            st.markdown(
                """
                <div class="status-box">
                    GREATER THAN 18
                </div>
                """,
                unsafe_allow_html=True
            )

        elif result == "LESS THAN 18":

            st.markdown(
                """
                <div class="status-box">
                    LESS THAN 18
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        # Status outside the camera
        with state.lock:

            started = state.verification_started
            capture_requested = state.capture_requested

        if capture_requested:

            st.markdown(
                """
                <div class="status-box">
                    PROCESSING...
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# RUN PREDICTION WORKER
# ============================================================

prediction_worker()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="warning">
        ⚠️ This AI estimation is for demonstration purposes only
        and should not be used as a definitive age determination.
    </div>
    """,
    unsafe_allow_html=True
)
