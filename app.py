import streamlit as st
import cv2
import numpy as np
import time
import threading
import base64

import onnxruntime as ort

from PIL import Image
from huggingface_hub import hf_hub_download

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Age Verification",
    page_icon="🔐",
    layout="centered"
)


# ============================================================
# CUSTOM BACKGROUND IMAGE
# ============================================================

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

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.12);
    }

    .info-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .info-text {
        font-size: 16px;
        line-height: 1.7;
    }

    .camera-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .note {
        text-align: center;
        font-size: 14px;
        margin-top: 15px;
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

        <div class="info-title">
            How does it work?
        </div>

        <div class="info-text">
            1. Position your face inside the guide box.<br>
            2. Keep your face inside the frame for 10 seconds.<br>
            3. At the end, AI analyzes one final snapshot and verifies your age group.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CAMERA TITLE
# ============================================================

st.markdown(
    '<div class="camera-title">📷 Camera</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_REPO = "onnx-community/age-gender-prediction-ONNX"
MODEL_FILE = "onnx/model.onnx"


@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    return session


model_session = load_model()


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_image(face_crop):

    # BGR → RGB
    image = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    # Resize to model input size
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Convert to float
    image = image.astype(
        np.float32
    ) / 255.0

    # ImageNet normalization
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

    # HWC → CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(
        np.float32
    )


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(face_crop):

    input_tensor = preprocess_image(
        face_crop
    )

    input_name = model_session.get_inputs()[0].name

    outputs = model_session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    logits = outputs[0]

    estimated_age = int(
        round(
            float(logits[0][0])
        )
    )

    estimated_age = max(
        0,
        min(
            100,
            estimated_age
        )
    )

    return estimated_age


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# APPLICATION STATE
# ============================================================

class AppState:

    def __init__(self):

        self.lock = threading.Lock()

        self.frame_counter = 0

        self.last_face = None

        self.face_valid = False

        self.stable_start_time = None

        self.is_stable = False

        self.observation_start_time = None

        self.observation_complete = False

        self.final_snapshot = None

        self.snapshot_captured = False

        self.prediction_pending = False

        self.prediction_running = False

        self.final_status = None


state = AppState()


# ============================================================
# SETTINGS
# ============================================================

OBSERVATION_DURATION = 10.0

STABILITY_DURATION = 1.0

DETECTION_INTERVAL = 12

DETECTION_SCALE = 0.30


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    height, width = image.shape[:2]

    state.frame_counter += 1

    current_time = time.time()

    # --------------------------------------------------------
    # RESIZED IMAGE FOR FACE DETECTION
    # --------------------------------------------------------

    small_width = int(
        width * DETECTION_SCALE
    )

    small_height = int(
        height * DETECTION_SCALE
    )

    small_frame = cv2.resize(
        image,
        (
            small_width,
            small_height
        )
    )

    gray = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = []

    if (
        state.frame_counter %
        DETECTION_INTERVAL
        == 0
    ):

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=6,
            minSize=(35, 35)
        )

        faces = list(faces)

        with state.lock:

            # Exactly one face
            if len(faces) == 1:

                x, y, w, h = faces[0]

                # Convert back to original resolution
                x = int(
                    x / DETECTION_SCALE
                )

                y = int(
                    y / DETECTION_SCALE
                )

                w = int(
                    w / DETECTION_SCALE
                )

                h = int(
                    h / DETECTION_SCALE
                )

                # ------------------------------------------------
                # FACE SIZE CHECK
                # ------------------------------------------------

                face_width_ratio = (
                    w / width
                )

                size_valid = (
                    face_width_ratio >= 0.20
                )


                # ------------------------------------------------
                # FACE CENTER
                # ------------------------------------------------

                face_center_x = (
                    x + w / 2
                )

                face_center_y = (
                    y + h / 2
                )

                frame_center_x = (
                    width / 2
                )

                frame_center_y = (
                    height / 2
                )

                horizontal_offset = abs(
                    face_center_x -
                    frame_center_x
                ) / width

                vertical_offset = abs(
                    face_center_y -
                    frame_center_y
                ) / height

                centered = (
                    horizontal_offset <= 0.15
                    and
                    vertical_offset <= 0.15
                )


                # ------------------------------------------------
                # FACE INSIDE GUIDE
                # ------------------------------------------------

                guide_left = int(
                    width * 0.20
                )

                guide_right = int(
                    width * 0.80
                )

                guide_top = int(
                    height * 0.12
                )

                guide_bottom = int(
                    height * 0.88
                )

                inside_guide = (
                    x >= guide_left
                    and
                    y >= guide_top
                    and
                    x + w <= guide_right
                    and
                    y + h <= guide_bottom
                )


                # ------------------------------------------------
                # FINAL VALIDATION
                # ------------------------------------------------

                valid = (
                    size_valid
                    and
                    centered
                    and
                    inside_guide
                )


                if valid:

                    current_face = (
                        x,
                        y,
                        w,
                        h
                    )

                    # --------------------------------------------
                    # CHECK STABILITY
                    # --------------------------------------------

                    if state.last_face is None:

                        state.last_face = (
                            current_face
                        )

                        state.stable_start_time = (
                            current_time
                        )

                    else:

                        old_x, old_y, old_w, old_h = (
                            state.last_face
                        )

                        movement = (
                            abs(x - old_x)
                            +
                            abs(y - old_y)
                            +
                            abs(w - old_w)
                            +
                            abs(h - old_h)
                        )

                        if movement < 80:

                            if (
                                state.stable_start_time
                                is None
                            ):

                                state.stable_start_time = (
                                    current_time
                                )

                        else:

                            state.stable_start_time = (
                                current_time
                            )

                        state.last_face = (
                            current_face
                        )


                    state.face_valid = True


                    # --------------------------------------------
                    # STABLE FOR 1 SECOND
                    # --------------------------------------------

                    if (
                        state.stable_start_time
                        is not None
                        and
                        current_time -
                        state.stable_start_time
                        >= STABILITY_DURATION
                    ):

                        state.is_stable = True

                        # Start observation
                        if (
                            state.observation_start_time
                            is None
                            and
                            not state.observation_complete
                        ):

                            state.observation_start_time = (
                                current_time
                            )

                else:

                    state.face_valid = False

                    state.is_stable = False

                    state.stable_start_time = None

                    state.last_face = None


    # ========================================================
    # OBSERVATION TIMER
    # ========================================================

    with state.lock:

        if (
            state.observation_start_time
            is not None
            and
            not state.observation_complete
        ):

            elapsed = (
                current_time -
                state.observation_start_time
            )

            # ------------------------------------------------
            # 10 SECONDS COMPLETED
            # ------------------------------------------------

            if (
                elapsed >= OBSERVATION_DURATION
                and
                not state.snapshot_captured
            ):

                # Use last detected face
                if state.last_face is not None:

                    x, y, w, h = (
                        state.last_face
                    )

                    # --------------------------------------------
                    # ADD PADDING AROUND FACE
                    # --------------------------------------------

                    padding_x = int(
                        w * 0.20
                    )

                    padding_y = int(
                        h * 0.20
                    )

                    crop_x1 = max(
                        0,
                        x - padding_x
                    )

                    crop_y1 = max(
                        0,
                        y - padding_y
                    )

                    crop_x2 = min(
                        width,
                        x + w + padding_x
                    )

                    crop_y2 = min(
                        height,
                        y + h + padding_y
                    )

                    face_crop = image[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2
                    ].copy()

                    # --------------------------------------------
                    # EXACTLY ONE SNAPSHOT
                    # --------------------------------------------

                    state.final_snapshot = (
                        face_crop
                    )

                    state.snapshot_captured = True

                    state.prediction_pending = True

                    state.observation_complete = True


    # ========================================================
    # DRAW GUIDE BOX
    # ========================================================

    guide_left = int(
        width * 0.20
    )

    guide_right = int(
        width * 0.80
    )

    guide_top = int(
        height * 0.12
    )

    guide_bottom = int(
        height * 0.88
    )


    # ========================================================
    # GUIDE COLOR
    # ========================================================

    if state.observation_complete:

        # Processing / completed
        guide_color = (
            0,
            255,
            0
        )

    elif state.is_stable:

        guide_color = (
            0,
            255,
            0
        )

    elif state.face_valid:

        guide_color = (
            0,
            220,
            255
        )

    else:

        guide_color = (
            0,
            0,
            255
        )


    # ========================================================
    # DRAW GUIDE
    # ========================================================

    cv2.rectangle(
        image,
        (
            guide_left,
            guide_top
        ),
        (
            guide_right,
            guide_bottom
        ),
        guide_color,
        3
    )


    # ========================================================
    # STATUS TEXT
    # ========================================================

    with state.lock:

        if state.final_status is not None:

            display_status = (
                state.final_status
            )

        elif state.observation_complete:

            display_status = (
                "PROCESSING..."
            )

        elif (
            state.observation_start_time
            is not None
        ):

            elapsed = (
                current_time -
                state.observation_start_time
            )

            remaining = max(
                0,
                OBSERVATION_DURATION -
                elapsed
            )

            display_status = (
                f"KEEP YOUR FACE IN FRAME • "
                f"{remaining:.1f}s"
            )

        elif state.is_stable:

            display_status = (
                "FACE READY ✓"
            )

        elif state.face_valid:

            display_status = (
                "HOLD STILL..."
            )

        else:

            display_status = (
                "POSITION YOUR FACE INSIDE THE BOX"
            )


    # ========================================================
    # STATUS BOX
    # ========================================================

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65

    thickness = 2

    text_size = cv2.getTextSize(
        display_status,
        font,
        font_scale,
        thickness
    )[0]

    text_width = text_size[0]

    text_height = text_size[1]

    box_padding_x = 20

    box_padding_y = 15

    box_width = (
        text_width +
        box_padding_x * 2
    )

    box_height = (
        text_height +
        box_padding_y * 2
    )

    box_x = int(
        (width - box_width) / 2
    )

    box_y = (
        height -
        box_height -
        20
    )


    # Black filled status box
    cv2.rectangle(
        image,
        (
            box_x,
            box_y
        ),
        (
            box_x + box_width,
            box_y + box_height
        ),
        (
            0,
            0,
            0
        ),
        -1
    )


    # White status text
    cv2.putText(
        image,
        display_status,
        (
            box_x + box_padding_x,
            box_y +
            box_padding_y +
            text_height
        ),
        font,
        font_scale,
        (
            255,
            255,
            255
        ),
        thickness,
        cv2.LINE_AA
    )


    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


# ============================================================
# PREDICTION WORKER
# ============================================================

@st.fragment(run_every=0.2)
def prediction_worker():

    snapshot = None

    with state.lock:

        if (
            state.prediction_pending
            and
            not state.prediction_running
            and
            state.final_snapshot is not None
        ):

            snapshot = (
                state.final_snapshot.copy()
            )

            state.prediction_pending = False

            state.prediction_running = True


    # --------------------------------------------------------
    # MODEL RUNS OUTSIDE VIDEO CALLBACK
    # --------------------------------------------------------

    if snapshot is not None:

        try:

            estimated_age = predict_age(
                snapshot
            )

            # Only show age group
            if estimated_age >= 18:

                result = (
                    "GREATER THAN 18"
                )

            else:

                result = (
                    "LESS THAN 18"
                )

            with state.lock:

                state.final_status = result

                state.prediction_running = False

        except Exception:

            with state.lock:

                state.final_status = (
                    "VERIFICATION FAILED"
                )

                state.prediction_running = False


# ============================================================
# WEBRTC CAMERA
# ============================================================

ctx = webrtc_streamer(

    key="age-verification-camera",

    mode=WebRtcMode.SENDRECV,

    # --------------------------------------------------------
    # STUN SERVER
    # --------------------------------------------------------

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
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
    <div class="note">
        ⚠️ This AI system provides an automated age estimation
        and should not be considered a definitive identity or age document.
    </div>
    """,
    unsafe_allow_html=True
)
