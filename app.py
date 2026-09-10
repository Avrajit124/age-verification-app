import streamlit as st
import cv2
import numpy as np
import time
import threading
import base64
import onnxruntime as ort

from PIL import Image
from huggingface_hub import hf_hub_download
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
# UI
# ============================================================

st.markdown("""
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
    color: #666;
    margin-bottom: 30px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 25px;
}

.camera-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 10px;
}

.note {
    margin-top: 20px;
    font-size: 14px;
    color: #777;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">🔐 Age Verification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered real-time facial age estimation</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

st.subheader("How does it work?")

st.write(
    "Look directly at the camera while our AI analyzes your face."
)

st.write(
    "Keep your face inside the box for **10 seconds**."
)

st.write(
    "At the end of 10 seconds, one photo is captured internally "
    "and the AI determines whether you are **above or below 18**."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="camera-title">📷 Camera</div>',
    unsafe_allow_html=True
)


# ============================================================
# ONNX AGE MODEL
# ============================================================

MODEL_REPO = "onnx-community/age-gender-prediction-ONNX"
MODEL_FILE = "onnx/model.onnx"


@st.cache_resource
def load_age_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    session_options = ort.SessionOptions()

    # Keep ONNX CPU inference efficient
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    return session


age_model = load_age_model()


# ============================================================
# MODEL PREPROCESSING
# ============================================================

IMAGE_SIZE = 224

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_face(face_crop):

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = Image.fromarray(rgb)

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # NumPy
    x = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    # ImageNet normalization
    x = (
        x - MEAN
    ) / STD

    # HWC -> CHW
    x = np.transpose(
        x,
        (2, 0, 1)
    )

    return x.astype(np.float32)


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(face_crop):

    if face_crop is None:
        return None

    if face_crop.size == 0:
        return None

    try:

        processed = preprocess_face(
            face_crop
        )

        # Single image batch
        batch = np.expand_dims(
            processed,
            axis=0
        ).astype(np.float32)

        input_name = age_model.get_inputs()[0].name

        outputs = age_model.run(
            None,
            {
                input_name: batch
            }
        )

        logits = outputs[0]

        # First value = predicted age
        age = float(
            logits[0][0]
        )

        # Clamp
        age = max(
            0.0,
            min(100.0, age)
        )

        return age

    except Exception as e:

        print(
            "Age prediction error:",
            e
        )

        return None


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# APPLICATION STATE
#
# IMPORTANT:
# This is intentionally NOT cached.
#
# Therefore when Streamlit reloads the page, a completely
# fresh verification state is created.
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

        # Exactly ONE final snapshot
        self.final_snapshot = None

        self.snapshot_captured = False

        # Prediction state
        self.prediction_pending = False

        self.prediction_running = False

        self.final_status = None


# Fresh state for every Streamlit script run
state = AppState()


# ============================================================
# SETTINGS
# ============================================================

OBSERVATION_DURATION = 10.0

STABILITY_DURATION = 1.0

DETECTION_INTERVAL = 12

DETECTION_SCALE = 0.30

MAX_DETECTION_MISSES = 5


# ============================================================
# VIDEO FRAME CALLBACK
#
# IMPORTANT:
# NO MODEL INFERENCE HERE.
#
# This callback only:
# - receives camera frames
# - detects face occasionally
# - draws UI
# - starts timer
# - captures ONE final snapshot
#
# This keeps the camera smooth.
# ============================================================

def video_frame_callback(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    height, width = img.shape[:2]

    now = time.time()


    # ========================================================
    # GUIDE BOX
    # ========================================================

    guide_width = int(
        width * 0.55
    )

    guide_height = int(
        height * 0.72
    )

    guide_x1 = (
        width - guide_width
    ) // 2

    guide_y1 = (
        height - guide_height
    ) // 2

    guide_x2 = (
        guide_x1 + guide_width
    )

    guide_y2 = (
        guide_y1 + guide_height
    )


    # ========================================================
    # FACE DETECTION
    #
    # Once the 10-second observation is complete, we don't
    # need to keep detecting faces.
    # ========================================================

    with state.lock:

        observation_complete = (
            state.observation_complete
        )

        state.frame_counter += 1

        current_frame_count = (
            state.frame_counter
        )


    if not observation_complete:

        # Detect only every N frames
        if (
            current_frame_count %
            DETECTION_INTERVAL == 0
        ):

            # Small image for faster Haar detection
            small = cv2.resize(
                img,
                None,
                fx=DETECTION_SCALE,
                fy=DETECTION_SCALE
            )

            gray = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=6,
                minSize=(35, 35)
            )


            detected_faces = []

            scale_back = (
                1.0 / DETECTION_SCALE
            )

            for (x, y, w, h) in faces:

                detected_faces.append(
                    (
                        int(x * scale_back),
                        int(y * scale_back),
                        int(w * scale_back),
                        int(h * scale_back)
                    )
                )


            # =================================================
            # EXACTLY ONE FACE
            # =================================================

            with state.lock:

                if len(detected_faces) == 1:

                    face = detected_faces[0]

                    x, y, w, h = face


                    # =========================================
                    # FACE CENTER
                    # =========================================

                    face_center_x = (
                        x + w / 2
                    )

                    face_center_y = (
                        y + h / 2
                    )

                    guide_center_x = (
                        width / 2
                    )

                    guide_center_y = (
                        height / 2
                    )


                    center_tolerance_x = (
                        width * 0.15
                    )

                    center_tolerance_y = (
                        height * 0.15
                    )


                    centered = (

                        abs(
                            face_center_x -
                            guide_center_x
                        )
                        <=
                        center_tolerance_x

                        and

                        abs(
                            face_center_y -
                            guide_center_y
                        )
                        <=
                        center_tolerance_y
                    )


                    # =========================================
                    # FACE SIZE
                    # =========================================

                    face_ratio = (
                        w / width
                    )

                    large_enough = (
                        face_ratio >= 0.20
                    )


                    # =========================================
                    # FACE INSIDE GUIDE
                    # =========================================

                    inside_guide = (

                        x >= guide_x1

                        and

                        y >= guide_y1

                        and

                        x + w <= guide_x2

                        and

                        y + h <= guide_y2
                    )


                    valid_position = (

                        large_enough

                        and

                        centered

                        and

                        inside_guide
                    )


                    if valid_position:

                        state.last_face = face

                        state.face_valid = True


                        # Start stability timer
                        if (
                            state.stable_start_time
                            is None
                        ):

                            state.stable_start_time = now


                        stable_time = (
                            now -
                            state.stable_start_time
                        )


                        if (
                            stable_time >=
                            STABILITY_DURATION
                        ):

                            state.is_stable = True

                    else:

                        state.face_valid = False

                        state.stable_start_time = None

                        state.is_stable = False

                else:

                    state.face_valid = False

                    state.stable_start_time = None

                    state.is_stable = False


                    # Only clear last face when there are
                    # multiple/no faces
                    if len(detected_faces) != 1:

                        state.last_face = None


    # ========================================================
    # READ CURRENT STATE
    # ========================================================

    with state.lock:

        face_valid = state.face_valid

        is_stable = state.is_stable

        observation_start = (
            state.observation_start_time
        )

        observation_complete = (
            state.observation_complete
        )

        final_status = (
            state.final_status
        )

        last_face = (
            state.last_face
        )


    # ========================================================
    # START 10-SECOND OBSERVATION
    #
    # Only starts after face has been stable for 1 second.
    # ========================================================

    if (

        is_stable

        and

        not observation_complete

        and

        observation_start is None
    ):

        with state.lock:

            if (
                state.observation_start_time
                is None
            ):

                state.observation_start_time = now

                observation_start = now


    # ========================================================
    # 10-SECOND OBSERVATION
    # ========================================================

    with state.lock:

        observation_start = (
            state.observation_start_time
        )

        observation_complete = (
            state.observation_complete
        )

        last_face = (
            state.last_face
        )


    if (

        observation_start is not None

        and

        not observation_complete
    ):

        elapsed = (
            now -
            observation_start
        )

        remaining = max(
            0.0,
            OBSERVATION_DURATION -
            elapsed
        )


        # ====================================================
        # We don't capture anything during these 10 seconds.
        # ====================================================

        if remaining > 0:

            countdown_text = (
                f"KEEP YOUR FACE IN FRAME  •  "
                f"{remaining:.1f}s"
            )

            text_size = cv2.getTextSize(
                countdown_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )[0]

            text_x = (
                width -
                text_size[0]
            ) // 2

            text_y = (
                guide_y1 - 25
            )

            # Black background
            cv2.rectangle(
                img,
                (
                    text_x - 15,
                    text_y -
                    text_size[1] -
                    15
                ),
                (
                    text_x +
                    text_size[0] +
                    15,
                    text_y + 10
                ),
                (0, 0, 0),
                -1
            )

            # White text
            cv2.putText(
                img,
                countdown_text,
                (
                    text_x,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # 10 SECONDS FINISHED
        #
        # EXACTLY ONE PHOTO IS CAPTURED HERE.
        # ====================================================

        else:

            with state.lock:

                if not state.observation_complete:

                    state.observation_complete = True

                    # ----------------------------------------
                    # Capture exactly ONE internal snapshot
                    # ----------------------------------------

                    if (
                        last_face is not None

                        and

                        not state.snapshot_captured
                    ):

                        x, y, w, h = last_face


                        # Padding around face
                        pad_x = int(
                            w * 0.20
                        )

                        pad_y = int(
                            h * 0.20
                        )


                        crop_x1 = max(
                            0,
                            x - pad_x
                        )

                        crop_y1 = max(
                            0,
                            y - pad_y
                        )

                        crop_x2 = min(
                            width,
                            x + w + pad_x
                        )

                        crop_y2 = min(
                            height,
                            y + h + pad_y
                        )


                        face_crop = img[
                            crop_y1:crop_y2,
                            crop_x1:crop_x2
                        ].copy()


                        if (
                            face_crop.size > 0
                        ):

                            state.final_snapshot = (
                                face_crop
                            )

                            state.snapshot_captured = True

                            state.prediction_pending = True


    # ========================================================
    # GUIDE COLOR
    # ========================================================

    with state.lock:

        current_face_valid = (
            state.face_valid
        )

        current_stable = (
            state.is_stable
        )

        current_complete = (
            state.observation_complete
        )

        current_status = (
            state.final_status
        )

        prediction_pending = (
            state.prediction_pending
        )

        prediction_running = (
            state.prediction_running
        )


    if current_complete:

        guide_color = (
            0, 255, 0
        )

    elif current_face_valid and current_stable:

        guide_color = (
            0, 255, 0
        )

    elif current_face_valid:

        guide_color = (
            0, 200, 255
        )

    else:

        guide_color = (
            0, 0, 255
        )


    # ========================================================
    # DRAW GUIDE
    # ========================================================

    cv2.rectangle(
        img,
        (
            guide_x1,
            guide_y1
        ),
        (
            guide_x2,
            guide_y2
        ),
        guide_color,
        3
    )


    # ========================================================
    # STATUS TEXT
    # ========================================================

    if current_complete:

        if current_status is not None:

            result_text = (
                current_status
            )

        elif (
            prediction_pending
            or
            prediction_running
        ):

            result_text = (
                "PROCESSING..."
            )

        else:

            result_text = (
                "PROCESSING..."
            )

    elif current_stable:

        result_text = (
            "FACE READY ✓"
        )

    elif current_face_valid:

        result_text = (
            "HOLD STILL..."
        )

    else:

        result_text = (
            "POSITION YOUR FACE INSIDE THE BOX"
        )


    # ========================================================
    # PROMINENT STATUS BOX
    # ========================================================

    text_size = cv2.getTextSize(
        result_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        3
    )[0]


    text_x = (
        width -
        text_size[0]
    ) // 2

    text_y = (
        height - 35
    )


    # Black filled background
    cv2.rectangle(
        img,
        (
            text_x - 18,
            text_y -
            text_size[1] -
            18
        ),
        (
            text_x +
            text_size[0] +
            18,
            text_y + 12
        ),
        (0, 0, 0),
        -1
    )


    # White text
    cv2.putText(
        img,
        result_text,
        (
            text_x,
            text_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )


    # ========================================================
    # RETURN FRAME
    # ========================================================

    return frame.from_ndarray(
        img,
        format="bgr24"
    )


# ============================================================
# PREDICTION WORKER
#
# This runs OUTSIDE the WebRTC callback.
#
# Therefore ONNX inference cannot block the camera callback.
# ============================================================

@st.fragment(run_every=0.2)
def prediction_worker():

    snapshot = None


    with state.lock:

        if (

            state.prediction_pending

            and

            not state.prediction_running
        ):

            if state.final_snapshot is not None:

                snapshot = (
                    state.final_snapshot.copy()
                )

                state.prediction_pending = False

                state.prediction_running = True


    # ========================================================
    # RUN MODEL
    # ========================================================

    if snapshot is not None:

        estimated_age = predict_age(
            snapshot
        )


        with state.lock:

            if estimated_age is not None:

                # Numerical age is intentionally hidden.

                if estimated_age >= 18:

                    state.final_status = (
                        "GREATER THAN 18"
                    )

                else:

                    state.final_status = (
                        "LESS THAN 18"
                    )

            else:

                state.final_status = (
                    "UNABLE TO ESTIMATE"
                )


            state.prediction_running = False


# Start prediction worker
prediction_worker()


# ============================================================
# CAMERA
# ============================================================

webrtc_streamer(
    key="age-verification-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 640
            },
            "height": {
                "ideal": 480
            },
            "frameRate": {
                "ideal": 30
            }
        },
        "audio": False
    },
    async_processing=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="note">⚠️ Age estimation is an AI prediction '
    'and may not be completely accurate.</div>',
    unsafe_allow_html=True
)