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
        background = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url(
                "data:image/png;base64,{background}"
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
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .info {
        background: rgba(255,255,255,0.95);
        border-radius: 14px;
        padding: 18px 22px;
        color: #111111;
        margin-bottom: 25px;
    }

    .info-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .info-line {
        font-size: 15px;
        margin: 6px 0;
    }

    .camera-title {
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .result {
        background: #000000;
        color: #ffffff;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: 800;
        margin-top: 15px;
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
# INFORMATION
# ============================================================

st.markdown(
    """
    <div class="info">
        <div class="info-title">How does it work?</div>

        <div class="info-line">
            1. Keep your face inside the square.
        </div>

        <div class="info-line">
            2. A 10-second countdown starts automatically.
        </div>

        <div class="info-line">
            3. At 0, one image is captured and analyzed by AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STATE
# ============================================================

class AppState:

    def __init__(self):

        self.lock = threading.Lock()

        # Latest camera frame
        self.latest_frame = None

        # Current face position
        self.face_box = None

        # Face inside square
        self.face_inside = False

        # Countdown
        self.countdown_started = False
        self.countdown_start_time = None

        # One-time capture
        self.captured = False
        self.captured_face = None

        # Prediction
        self.prediction_started = False
        self.prediction_finished = False
        self.result = None

        # Detection optimization
        self.frame_count = 0
        self.last_detection_time = 0.0

        # Final state
        self.finished = False
        self.error = None


if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()

state = st.session_state.app_state


# ============================================================
# HAAR CASCADE
# ============================================================

cascade_path = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    cascade_path
)


# ============================================================
# ONNX MODEL
# ============================================================

MODEL_REPO = "onnx-community/age-gender-prediction-ONNX"
MODEL_FILE = "onnx/model.onnx"


@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )


# ============================================================
# FACE DETECTION
# ============================================================

def find_face(frame):

    if frame is None:
        return None

    # Small image = faster detection
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

    # Largest face
    x, y, w, h = max(
        faces,
        key=lambda r: r[2] * r[3]
    )

    # Convert coordinates to original frame
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    return x, y, w, h


# ============================================================
# CHECK FACE INSIDE SQUARE
# ============================================================

def face_inside_square(
    box,
    square
):

    if box is None:
        return False

    x, y, w, h = box

    sx, sy, sw, sh = square

    # Entire face must be inside square
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
# CROP FINAL FACE
# ============================================================

def crop_face(
    frame,
    box
):

    if frame is None or box is None:
        return None

    x, y, w, h = box

    frame_h, frame_w = frame.shape[:2]

    # Slight margin
    mx = int(w * 0.20)
    my = int(h * 0.25)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)

    x2 = min(
        frame_w,
        x + w + mx
    )

    y2 = min(
        frame_h,
        y + h + my
    )

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face.copy()


# ============================================================
# MODEL PREPROCESSING
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

    image /= 255.0

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

    input_name = model.get_inputs()[0].name

    tensor = preprocess(face)

    output = model.run(
        None,
        {
            input_name: tensor
        }
    )

    age = float(
        np.asarray(
            output[0]
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
    # SQUARE
    # ========================================================

    # One single square
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
    # SAVE LATEST FRAME
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

    # Do detection only when necessary.
    # Once countdown starts, we don't need to repeatedly
    # process/capture frames for age prediction.

    if not countdown_started and not captured and not finished:

        # Detect roughly every 10 frames
        if frame_number % 10 == 0:

            box = find_face(image)

            inside = face_inside_square(
                box,
                square
            )

            with state.lock:

                state.face_box = box
                state.face_inside = inside
                state.last_detection_time = now

    else:

        with state.lock:

            box = state.face_box
            inside = state.face_inside

    # ========================================================
    # READ CURRENT FACE STATE
    # ========================================================

    with state.lock:

        box = state.face_box
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

            countdown_started = True

            countdown_start = now

    # ========================================================
    # COUNTDOWN
    # ========================================================

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

        remaining = 10.0 - elapsed

        # ====================================================
        # 10 → 1
        # ====================================================

        if remaining > 0:

            countdown_number = int(
                np.ceil(remaining)
            )

        # ====================================================
        # 0
        # ====================================================

        else:

            with state.lock:

                if (
                    not state.captured
                    and
                    state.latest_frame is not None
                    and
                    state.face_box is not None
                ):

                    # ========================================
                    # ONLY ONE PHOTO AT ZERO
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

                        state.captured_face = final_face

                        state.captured = True

                        state.countdown_started = False

    # ========================================================
    # DRAW VIDEO
    # ========================================================

    display = image.copy()

    # Re-read state
    with state.lock:

        box = state.face_box
        inside = state.face_inside
        countdown_started = (
            state.countdown_started
        )
        countdown_start = (
            state.countdown_start_time
        )
        captured = state.captured
        finished = state.finished
        result = state.result

    # ========================================================
    # SQUARE COLOR
    # ========================================================

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

    # ========================================================
    # DRAW ONLY ONE SQUARE
    # ========================================================

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
    # STATUS
    # ========================================================

    if finished and result:

        status = result

    elif captured:

        status = "PROCESSING..."

    elif countdown_started and countdown_start:

        elapsed = (
            now - countdown_start
        )

        remaining = max(
            0,
            10.0 - elapsed
        )

        countdown_number = int(
            np.ceil(remaining)
        )

        status = str(
            countdown_number
        )

    else:

        status = (
            "PLEASE KEEP YOUR FACE INSIDE THE SQUARE"
        )

    # ========================================================
    # COUNTDOWN NUMBER
    # ========================================================

    if (
        countdown_started
        and
        not captured
        and
        not finished
    ):

        text = status

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 2.8

        thickness = 6

        text_size = cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )[0]

        text_x = (
            frame_w - text_size[0]
        ) // 2

        text_y = (
            frame_h + text_size[1]
        ) // 2

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
            text,
            (
                text_x,
                text_y
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    else:

        # ====================================================
        # NORMAL MESSAGE
        # ====================================================

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.65

        thickness = 2

        text_size = cv2.getTextSize(
            status,
            font,
            scale,
            thickness
        )[0]

        text_x = (
            frame_w - text_size[0]
        ) // 2

        text_y = frame_h - 30

        padding_x = 18
        padding_y = 12

        cv2.rectangle(
            display,
            (
                text_x - padding_x,
                text_y - text_size[1] - padding_y
            ),
            (
                text_x + text_size[0] + padding_x,
                text_y + padding_y
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            status,
            (
                text_x,
                text_y
            ),
            font,
            scale,
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
# WEBRTC
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
# PREDICTION
# ============================================================

@st.fragment(run_every=0.25)
def prediction_worker():

    with state.lock:

        captured = state.captured

        prediction_started = (
            state.prediction_started
        )

        prediction_finished = (
            state.prediction_finished
        )

        captured_face = (
            state.captured_face
        )

        result = state.result

        error = state.error

    # ========================================================
    # RUN MODEL EXACTLY ONCE
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

            # ================================================
            # MODEL ONLY RUNS HERE
            # ================================================

            age = predict_age(
                captured_face
            )

            if age >= 18:

                result = "GREATER THAN 18"

            else:

                result = "LESS THAN 18"

            with state.lock:

                state.result = result

                state.prediction_finished = True

                state.finished = True

        except Exception as e:

            with state.lock:

                state.error = str(e)

                state.prediction_finished = True

                state.finished = True

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    with state.lock:

        result = state.result

        finished = state.finished

        error = state.error

    if error:

        st.error(
            "Unable to analyze the captured face."
        )

    elif finished and result:

        st.markdown(
            f"""
            <div class="result">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif captured:

        st.markdown(
            """
            <div class="result">
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
