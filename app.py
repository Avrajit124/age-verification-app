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
        font-size: 22px;
        font-weight: 800;
        background: #000000;
        color: #ffffff;
    }

    .countdown-box {
        text-align: center;
        font-size: 42px;
        font-weight: 900;
        margin-top: 12px;
        padding: 8px;
        border-radius: 12px;
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
# INFO CARD
# ============================================================

st.markdown(
    """
    <div class="info-card">

        <div class="info-title">How does it work?</div>

        <div class="info-line">
            1️⃣ Position your face inside the guide box.
        </div>

        <div class="info-line">
            2️⃣ Keep your face steady during the 10-second verification.
        </div>

        <div class="info-line">
            3️⃣ At 0 seconds, one photo is captured and analyzed by AI.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RUNTIME STATE
# ============================================================

class VerificationState:

    def __init__(self):

        self.lock = threading.Lock()

        # ----------------------------
        # Face
        # ----------------------------

        self.face_box = None
        self.face_valid = False
        self.last_face_time = 0.0

        # ----------------------------
        # Stability
        # ----------------------------

        self.stable_start = None

        # ----------------------------
        # Countdown
        # ----------------------------

        self.verification_started = False
        self.verification_start_time = None

        # ----------------------------
        # Capture
        # ----------------------------

        self.capture_requested = False
        self.capture_done = False
        self.captured_face = None

        # ----------------------------
        # Prediction
        # ----------------------------

        self.prediction_started = False
        self.prediction_done = False
        self.prediction_result = None

        # ----------------------------
        # Camera
        # ----------------------------

        self.last_frame = None
        self.frame_counter = 0

        # ----------------------------
        # Final
        # ----------------------------

        self.finished = False
        self.error = None


if "verification_state" not in st.session_state:

    st.session_state.verification_state = VerificationState()

state = st.session_state.verification_state


# ============================================================
# FACE DETECTOR
# ============================================================

CASCADE_PATH = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


# ============================================================
# LOAD ONNX MODEL
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

    if frame is None:
        return None

    height, width = frame.shape[:2]

    # Small image for faster detection
    scale = 0.35

    small_frame = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        small_frame,
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

    # Largest detected face
    x, y, w, h = max(
        faces,
        key=lambda r: r[2] * r[3]
    )

    # Convert back to original frame coordinates
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

    frame_h, frame_w = frame.shape[:2]

    x, y, w, h = box

    if w <= 0 or h <= 0:
        return False

    # Face should occupy reasonable portion of frame
    face_ratio = (w * h) / float(
        frame_w * frame_h
    )

    if face_ratio < 0.035:
        return False

    # Don't allow face touching edges
    margin_x = int(frame_w * 0.04)
    margin_y = int(frame_h * 0.04)

    if x < margin_x:
        return False

    if y < margin_y:
        return False

    if x + w > frame_w - margin_x:
        return False

    if y + h > frame_h - margin_y:
        return False

    return True


# ============================================================
# CROP FACE
# ============================================================

def crop_face(frame, box):

    if frame is None or box is None:
        return None

    x, y, w, h = box

    frame_h, frame_w = frame.shape[:2]

    # Extra margin around face
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.25)

    x1 = max(
        0,
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        frame_w,
        x + w + margin_x
    )

    y2 = min(
        frame_h,
        y + h + margin_y
    )

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face.copy()


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def prepare_face(face):

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

    image = image / 255.0

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
# MODEL PREDICTION
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

    age = float(
        np.asarray(outputs[0]).reshape(-1)[0]
    )

    return age


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    # Convert incoming frame
    image = frame.to_ndarray(
        format="bgr24"
    )

    now = time.monotonic()

    # ========================================================
    # UPDATE FRAME STATE
    # ========================================================

    with state.lock:

        state.frame_counter += 1

        frame_number = state.frame_counter

        # Always keep latest frame.
        # This is used ONLY for the final single capture.
        state.last_frame = image.copy()

    # ========================================================
    # FACE DETECTION
    # ========================================================

    # Detection is intentionally not done on every frame.
    # This keeps the actual camera feed smooth.

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

        # Keep the last detection briefly between
        # actual detection frames.

        with state.lock:

            if (
                state.last_face_time > 0
                and
                now - state.last_face_time < 0.7
            ):

                state.face_valid = True

            else:

                state.face_valid = False

    # ========================================================
    # READ STATE
    # ========================================================

    with state.lock:

        box = state.face_box
        valid = state.face_valid
        started = state.verification_started
        start_time = state.verification_start_time
        capture_done = state.capture_done
        finished = state.finished

    # ========================================================
    # START COUNTDOWN
    # ========================================================

    if (
        not started
        and
        not finished
        and
        not capture_done
    ):

        if valid:

            with state.lock:

                if state.stable_start is None:

                    state.stable_start = now

                stable_time = (
                    now - state.stable_start
                )

                # Face must remain detected for 1 second
                if stable_time >= 1.0:

                    # ========================================
                    # START EXACT 10 SECOND COUNTDOWN
                    # ========================================

                    state.verification_started = True

                    state.verification_start_time = now

                    state.capture_requested = False

                    state.capture_done = False

                    state.prediction_started = False

                    state.prediction_done = False

                    state.prediction_result = None

        else:

            with state.lock:

                state.stable_start = None

    # ========================================================
    # 10 → 0 COUNTDOWN
    # ========================================================

    with state.lock:

        started = state.verification_started
        start_time = state.verification_start_time
        capture_done = state.capture_done
        current_valid = state.face_valid
        current_box = state.face_box

    if (
        started
        and
        start_time is not None
        and
        not capture_done
    ):

        elapsed = now - start_time

        remaining = 10.0 - elapsed

        # ====================================================
        # COUNTDOWN: 10, 9, 8 ... 1
        # ====================================================

        if remaining > 0:

            countdown_number = int(
                np.ceil(remaining)
            )

        # ====================================================
        # ZERO REACHED
        # ====================================================

        else:

            with state.lock:

                # =================================================
                # EXACTLY ONE CAPTURE
                # =================================================

                if (
                    not state.capture_done
                    and
                    state.last_frame is not None
                    and
                    state.face_box is not None
                    and
                    state.face_valid
                ):

                    captured = crop_face(
                        state.last_frame,
                        state.face_box
                    )

                    if captured is not None:

                        # Save ONE face image in memory
                        state.captured_face = captured

                        # Tell prediction worker
                        state.capture_requested = True

                        # Never capture again
                        state.capture_done = True


    # ========================================================
    # DRAW VIDEO
    # ========================================================

    display = image.copy()

    frame_h, frame_w = display.shape[:2]

    with state.lock:

        box = state.face_box
        valid = state.face_valid
        started = state.verification_started
        start_time = state.verification_start_time
        capture_requested = state.capture_requested
        finished = state.finished
        prediction_result = state.prediction_result

    # ========================================================
    # GUIDE BOX
    # ========================================================

    guide_width = int(
        frame_w * 0.52
    )

    guide_height = int(
        frame_h * 0.70
    )

    guide_x = (
        frame_w - guide_width
    ) // 2

    guide_y = (
        frame_h - guide_height
    ) // 2

    if finished:

        guide_color = (
            0,
            255,
            0
        )

    elif valid:

        guide_color = (
            0,
            255,
            0
        )

    else:

        guide_color = (
            0,
            0,
            255
        )

    cv2.rectangle(
        display,
        (
            guide_x,
            guide_y
        ),
        (
            guide_x + guide_width,
            guide_y + guide_height
        ),
        guide_color,
        3
    )

    # ========================================================
    # FACE BOX
    # ========================================================

    if box is not None:

        x, y, w, h = box

        cv2.rectangle(
            display,
            (
                x,
                y
            ),
            (
                x + w,
                y + h
            ),
            guide_color,
            2
        )

    # ========================================================
    # STATUS
    # ========================================================

    if finished:

        if prediction_result:

            status_text = prediction_result

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

        # ----------------------------------------------------
        # Reverse countdown
        # ----------------------------------------------------

        countdown_number = int(
            np.ceil(remaining)
        )

        status_text = str(
            countdown_number
        )

    elif valid:

        status_text = "FACE READY ✓"

    else:

        status_text = (
            "POSITION YOUR FACE INSIDE THE BOX"
        )

    # ========================================================
    # COUNTDOWN / STATUS DISPLAY
    # ========================================================

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Large countdown
    if (
        started
        and
        not capture_requested
        and
        not finished
    ):

        text_size = cv2.getTextSize(
            status_text,
            font,
            2.2,
            5
        )[0]

        text_x = (
            frame_w - text_size[0]
        ) // 2

        text_y = (
            frame_h + text_size[1]
        ) // 2

        # Black background
        padding = 30

        cv2.rectangle(
            display,
            (
                text_x - padding,
                text_y - text_size[1] - padding
            ),
            (
                text_x + text_size[0] + padding,
                text_y + padding
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            status_text,
            (
                text_x,
                text_y
            ),
            font,
            2.2,
            (255, 255, 255),
            5,
            cv2.LINE_AA
        )

    else:

        # Normal status box
        font_scale = 0.70
        thickness = 2

        text_size = cv2.getTextSize(
            status_text,
            font,
            font_scale,
            thickness
        )[0]

        text_x = (
            frame_w - text_size[0]
        ) // 2

        text_y = frame_h - 35

        padding_x = 20
        padding_y = 12

        rect_x1 = max(
            10,
            text_x - padding_x
        )

        rect_y1 = max(
            10,
            text_y - text_size[1] - padding_y
        )

        rect_x2 = min(
            frame_w - 10,
            text_x + text_size[0] + padding_x
        )

        rect_y2 = min(
            frame_h - 10,
            text_y + padding_y
        )

        cv2.rectangle(
            display,
            (
                rect_x1,
                rect_y1
            ),
            (
                rect_x2,
                rect_y2
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            status_text,
            (
                text_x,
                text_y
            ),
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
# CAMERA TITLE
# ============================================================

st.markdown(
    '<div class="camera-title">📷 Camera</div>',
    unsafe_allow_html=True
)


# ============================================================
# WEBRTC CAMERA
# ============================================================

ctx = webrtc_streamer(

    # IMPORTANT:
    # Fixed key prevents unnecessary component recreation.
    key="age-verification-camera",

    mode=WebRtcMode.SENDRECV,

    # ========================================================
    # STUN
    # ========================================================

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

    # ========================================================
    # CAMERA SETTINGS
    # ========================================================

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    # Keep video processing asynchronous
    async_processing=True,

    video_frame_callback=video_frame_callback
)


# ============================================================
# PREDICTION WORKER
# ============================================================

@st.fragment(run_every=0.25)
def prediction_worker():

    with state.lock:

        capture_requested = (
            state.capture_requested
        )

        prediction_started = (
            state.prediction_started
        )

        prediction_done = (
            state.prediction_done
        )

        captured_face = (
            state.captured_face
        )

        finished = state.finished

        result = state.prediction_result

        error = state.error

    # ========================================================
    # MODEL RUNS ONLY AFTER ONE CAPTURE
    # ========================================================

    if (
        capture_requested
        and
        not prediction_started
        and
        not prediction_done
        and
        captured_face is not None
    ):

        with state.lock:

            # Prevent another prediction
            state.prediction_started = True

        try:

            # =================================================
            # MODEL RUNS HERE — NOT IN VIDEO CALLBACK
            # =================================================

            age = predict_age(
                captured_face
            )

            # =================================================
            # RESULT
            # =================================================

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
    # READ FINAL STATE
    # ========================================================

    with state.lock:

        result = state.prediction_result

        finished = state.finished

        error = state.error

    # ========================================================
    # DISPLAY RESULT BELOW CAMERA
    # ========================================================

    if error:

        st.error(
            "Unable to analyze the captured face."
        )

    elif finished and result:

        st.markdown(
            f"""
            <div class="status-box">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif capture_requested:

        st.markdown(
            """
            <div class="status-box">
                PROCESSING...
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# START PREDICTION WORKER
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
