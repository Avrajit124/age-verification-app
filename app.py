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

if os.path.exists("background.png"):

    with open("background.png", "rb") as f:
        background_data = base64.b64encode(
            f.read()
        ).decode()

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
# UI CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .instruction {
        background: rgba(255, 255, 255, 0.95);
        color: #111111;
        padding: 18px 20px;
        border-radius: 14px;
        text-align: center;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 25px;
    }

    .camera-title {
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .message {
        text-align: center;
        background: #000000;
        color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 700;
        margin-top: 10px;
    }

    .countdown {
        text-align: center;
        background: #000000;
        color: #ffffff;
        padding: 8px;
        border-radius: 12px;
        font-size: 42px;
        font-weight: 900;
        margin-top: 10px;
    }

    .processing {
        text-align: center;
        background: #000000;
        color: #ffffff;
        padding: 13px;
        border-radius: 10px;
        font-size: 21px;
        font-weight: 800;
        margin-top: 10px;
    }

    .result {
        text-align: center;
        background: #000000;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        font-size: 25px;
        font-weight: 900;
        margin-top: 12px;
    }

    .warning {
        text-align: center;
        font-size: 13px;
        margin-top: 18px;
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
    '<div class="title">🔐 Age Verification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered real-time facial age estimation</div>',
    unsafe_allow_html=True
)


# ============================================================
# ONLY INSTRUCTION
# ============================================================

st.markdown(
    """
    <div class="instruction">
        Click on the Start button and place your face on the square.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION STATE
# ============================================================

class AppState:

    def __init__(self):

        self.lock = threading.Lock()

        # Latest camera frame
        self.latest_frame = None

        # Face
        self.face_box = None
        self.face_inside = False

        # Detection
        self.frame_count = 0

        # Countdown
        self.countdown_started = False
        self.countdown_start_time = None

        # One final capture
        self.captured = False
        self.captured_face = None

        # Prediction
        self.prediction_started = False
        self.prediction_finished = False
        self.result = None

        # Final
        self.finished = False
        self.error = None


if "app_state" not in st.session_state:

    st.session_state.app_state = AppState()


state = st.session_state.app_state


# ============================================================
# FACE DETECTOR
# ============================================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    cascade_path
)


# ============================================================
# MODEL
# ============================================================

MODEL_REPO = (
    "onnx-community/age-gender-prediction-ONNX"
)

MODEL_FILE = "onnx/model.onnx"


@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    return session


# ============================================================
# FIND FACE
# ============================================================

def find_face(frame):

    if frame is None:
        return None

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

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda r: r[2] * r[3]
    )

    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    return x, y, w, h


# ============================================================
# CHECK FACE INSIDE SQUARE
# ============================================================

def is_face_inside_square(
    face_box,
    square
):

    if face_box is None:
        return False

    x, y, w, h = face_box

    sx, sy, sw, sh = square

    return (
        x >= sx
        and
        y >= sy
        and
        x + w <= sx + sw
        and
        y + h <= sy + sh
    )


# ============================================================
# CROP FACE
# ============================================================

def crop_face(
    frame,
    box
):

    if frame is None or box is None:
        return None

    x, y, w, h = box

    frame_h, frame_w = frame.shape[:2]

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

    face = frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    return face.copy()


# ============================================================
# PREPROCESS MODEL INPUT
# ============================================================

def preprocess(face):

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

    image = (
        image - mean
    ) / std

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(
        np.float32
    )


# ============================================================
# PREDICT AGE
# ============================================================

def predict_age(face):

    model = load_model()

    input_name = (
        model.get_inputs()[0].name
    )

    tensor = preprocess(face)

    outputs = model.run(
        None,
        {
            input_name: tensor
        }
    )

    age = float(
        np.asarray(
            outputs[0]
        ).reshape(-1)[0]
    )

    return age


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    now = time.monotonic()

    frame_h, frame_w = image.shape[:2]

    # ========================================================
    # ONE SQUARE
    # ========================================================

    square_size = int(
        min(frame_w, frame_h) * 0.62
    )

    square_x = (
        frame_w - square_size
    ) // 2

    square_y = (
        frame_h - square_size
    ) // 2

    square = (
        square_x,
        square_y,
        square_size,
        square_size
    )

    # ========================================================
    # STORE LATEST FRAME
    # ========================================================

    with state.lock:

        state.latest_frame = image.copy()

        state.frame_count += 1

        frame_number = state.frame_count

        countdown_started = (
            state.countdown_started
        )

        captured = state.captured

        finished = state.finished

    # ========================================================
    # FACE DETECTION
    # ========================================================

    # Only detect when we are still waiting for the face
    # or checking whether the face remains inside.
    #
    # NO AGE MODEL HERE.
    #
    # NO PHOTO CAPTURE HERE.

    if not captured and not finished:

        if frame_number % 8 == 0:

            detected_face = find_face(
                image
            )

            inside = is_face_inside_square(
                detected_face,
                square
            )

            with state.lock:

                state.face_box = detected_face

                state.face_inside = inside

    # ========================================================
    # READ FACE STATUS
    # ========================================================

    with state.lock:

        inside = state.face_inside

        countdown_started = (
            state.countdown_started
        )

        countdown_start = (
            state.countdown_start_time
        )

        captured = state.captured

        finished = state.finished

    # ========================================================
    # START COUNTDOWN IMMEDIATELY
    # ========================================================

    if (
        inside
        and
        not countdown_started
        and
        not captured
        and
        not finished
    ):

        with state.lock:

            state.countdown_started = True

            state.countdown_start_time = now

    # ========================================================
    # IF FACE LEAVES SQUARE → RESET COUNTDOWN
    # ========================================================

    elif (
        not inside
        and
        countdown_started
        and
        not captured
        and
        not finished
    ):

        with state.lock:

            state.countdown_started = False

            state.countdown_start_time = None

    # ========================================================
    # COUNTDOWN
    # ========================================================

    with state.lock:

        countdown_started = (
            state.countdown_started
        )

        countdown_start = (
            state.countdown_start_time
        )

        captured = state.captured

    if (
        countdown_started
        and
        countdown_start is not None
        and
        not captured
    ):

        elapsed = (
            now - countdown_start
        )

        remaining = (
            10.0 - elapsed
        )

        # ====================================================
        # STILL COUNTING
        # ====================================================

        if remaining > 0:

            pass

        # ====================================================
        # 0 REACHED
        # ====================================================

        else:

            with state.lock:

                if (
                    not state.captured
                    and
                    state.latest_frame is not None
                    and
                    state.face_box is not None
                    and
                    state.face_inside
                ):

                    # ========================================
                    # ONLY ONE FINAL CAPTURE
                    # ========================================

                    final_frame = (
                        state.latest_frame.copy()
                    )

                    final_box = state.face_box

                    final_face = crop_face(
                        final_frame,
                        final_box
                    )

                    if final_face is not None:

                        state.captured_face = (
                            final_face
                        )

                        state.captured = True

                        state.countdown_started = False

                        state.countdown_start_time = None

    # ========================================================
    # DRAW ONLY THE SQUARE
    # ========================================================

    display = image.copy()

    with state.lock:

        inside = state.face_inside
        finished = state.finished

    # Green if face is inside
    # Red otherwise

    if inside or finished:

        square_color = (
            0,
            255,
            0
        )

    else:

        square_color = (
            0,
            0,
            255
        )

    cv2.rectangle(
        display,
        (
            square_x,
            square_y
        ),
        (
            square_x + square_size,
            square_y + square_size
        ),
        square_color,
        4
    )

    # ========================================================
    # IMPORTANT:
    #
    # NO TEXT IS DRAWN INSIDE THE CAMERA.
    #
    # Therefore the face remains completely visible.
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
# CAMERA
# ============================================================

webrtc_streamer(

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

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True,

    video_frame_callback=video_frame_callback
)


# ============================================================
# STATUS / COUNTDOWN / PREDICTION
# ============================================================

@st.fragment(run_every=0.15)
def display_status():

    with state.lock:

        inside = state.face_inside

        countdown_started = (
            state.countdown_started
        )

        countdown_start = (
            state.countdown_start_time
        )

        captured = state.captured

        prediction_started = (
            state.prediction_started
        )

        prediction_finished = (
            state.prediction_finished
        )

        result = state.result

        finished = state.finished

        captured_face = (
            state.captured_face
        )

        error = state.error

    # ========================================================
    # RUN MODEL ONLY AFTER FINAL CAPTURE
    # ========================================================

    if (
        captured
        and
        not prediction_started
        and
        not prediction_finished
        and
        captured_face is not None
    ):

        with state.lock:

            state.prediction_started = True

        try:

            # =================================================
            # MODEL RUNS ONLY ONCE
            # =================================================

            age = predict_age(
                captured_face
            )

            if age >= 18:

                final_result = (
                    "GREATER THAN 18"
                )

            else:

                final_result = (
                    "LESS THAN 18"
                )

            with state.lock:

                state.result = final_result

                state.prediction_finished = True

                state.finished = True

        except Exception as e:

            with state.lock:

                state.error = str(e)

                state.prediction_finished = True

                state.finished = True

    # ========================================================
    # READ UPDATED RESULT
    # ========================================================

    with state.lock:

        inside = state.face_inside

        countdown_started = (
            state.countdown_started
        )

        countdown_start = (
            state.countdown_start_time
        )

        captured = state.captured

        result = state.result

        finished = state.finished

        error = state.error

    # ========================================================
    # ERROR
    # ========================================================

    if error:

        st.markdown(
            """
            <div class="processing">
                Unable to analyze the face.
            </div>
            """,
            unsafe_allow_html=True
        )

        return

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if finished and result:

        st.markdown(
            f"""
            <div class="result">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        return

    # ========================================================
    # PROCESSING
    # ========================================================

    if captured:

        st.markdown(
            """
            <div class="processing">
                PROCESSING...
            </div>
            """,
            unsafe_allow_html=True
        )

        return

    # ========================================================
    # COUNTDOWN
    # ========================================================

    if (
        countdown_started
        and
        countdown_start is not None
    ):

        elapsed = (
            time.monotonic()
            - countdown_start
        )

        remaining = max(
            0.0,
            10.0 - elapsed
        )

        countdown_number = int(
            np.ceil(remaining)
        )

        # Prevent showing 0 here.
        # At 0 the callback captures the face
        # and this changes to PROCESSING.

        if countdown_number > 0:

            st.markdown(
                f"""
                <div class="countdown">
                    {countdown_number}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="processing">
                    PROCESSING...
                </div>
                """,
                unsafe_allow_html=True
            )

        return

    # ========================================================
    # FACE OUTSIDE SQUARE
    # ========================================================

    if not inside:

        st.markdown(
            """
            <div class="message">
                PLEASE KEEP YOUR FACE INSIDE THE SQUARE
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# START STATUS LOOP
# ============================================================

display_status()


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
