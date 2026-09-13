import streamlit as st
import cv2
import tempfile
import os
import random
from pathlib import Path

from ultralytics import YOLO
import gdown


st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide",
)


st.title("🚧 Road Damage Detection")
st.caption("AI-powered road damage analysis using a trained YOLO model")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = Path("/tmp/best.pt")

# Google Drive file containing the trained best.pt model
DRIVE_FILE_ID = "1B0XF-Mnn62Wrv3G9IPeMaavkpO6ReAyl"


# ============================================================
# DEMO EVENT DATA
# ============================================================

INCIDENT_EXAMPLES = [
    {
        "eventType": "POTHOLE",
        "busId": "BUS_TEST_02",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-13T16:15:00Z",
        "location": {
            "latitude": 28.6125,
            "longitude": 77.2250,
            "address": "Barakhamba Road, New Delhi"
        },
        "detection": {
            "confidence": 0.96,
            "severity": "HIGH"
        },
        "model": {
            "name": "pothole-yolo-test",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "test": True,
            "source": "streamlit-demo"
        }
    },
    {
        "eventType": "ROAD_DAMAGE",
        "busId": "BUS_TEST_01",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-13T17:22:10Z",
        "location": {
            "latitude": 28.6139,
            "longitude": 77.2090,
            "address": "Connaught Place, New Delhi"
        },
        "detection": {
            "confidence": 0.91,
            "severity": "MEDIUM"
        },
        "model": {
            "name": "road-damage-yolo",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "test": True,
            "source": "streamlit-demo"
        }
    },
    {
        "eventType": "POTHOLE",
        "busId": "BUS_TEST_03",
        "cameraId": "CAM_LEFT",
        "timestamp": "2026-09-13T18:05:42Z",
        "location": {
            "latitude": 28.6304,
            "longitude": 77.2177,
            "address": "Mandi House, New Delhi"
        },
        "detection": {
            "confidence": 0.88,
            "severity": "MEDIUM"
        },
        "model": {
            "name": "pothole-yolo-test",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "test": True,
            "source": "streamlit-demo"
        }
    },
    {
        "eventType": "ROAD_HAZARD",
        "busId": "BUS_TEST_04",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-13T19:14:25Z",
        "location": {
            "latitude": 28.6289,
            "longitude": 77.2065,
            "address": "India Gate, New Delhi"
        },
        "detection": {
            "confidence": 0.93,
            "severity": "HIGH"
        },
        "model": {
            "name": "road-damage-yolo",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "test": True,
            "source": "streamlit-demo"
        }
    },
    {
        "eventType": "POTHOLE",
        "busId": "BUS_TEST_05",
        "cameraId": "CAM_RIGHT",
        "timestamp": "2026-09-13T20:31:08Z",
        "location": {
            "latitude": 28.6416,
            "longitude": 77.1210,
            "address": "Rajouri Garden, New Delhi"
        },
        "detection": {
            "confidence": 0.97,
            "severity": "HIGH"
        },
        "model": {
            "name": "pothole-yolo-test",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "test": True,
            "source": "streamlit-demo"
        }
    }
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Download model only once
    if not MODEL_PATH.exists():

        with st.spinner("Downloading trained road-damage model..."):

            downloaded = gdown.download(
                id=DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False
            )

        if downloaded is None or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Make sure the Google Drive "
                "file is shared as 'Anyone with the link → Viewer'."
            )

    return YOLO(str(MODEL_PATH))


try:
    model = load_model()

    st.success("✅ Trained YOLO model loaded successfully.")

except Exception as e:

    st.error(f"❌ Could not load the model: {e}")
    st.stop()


# ============================================================
# SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Minimum confidence",
        0.10,
        0.90,
        0.35,
        0.05
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640],
        value=640
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Upload a road / accident video",
    type=["mp4", "avi", "mov", "mkv"]
)


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    input_path,
    output_path,
    confidence,
    image_size
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the uploaded video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create the output video."
        )

    progress = st.progress(0)
    status = st.empty()

    frame_number = 0
    detection_count = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=image_size,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                detection_count += len(
                    result.boxes
                )

            annotated_frame = result.plot()

            out.write(
                annotated_frame
            )

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number / total_frames,
                        1.0
                    )
                )

                status.text(
                    f"Processing frame "
                    f"{frame_number} / "
                    f"{total_frames} "
                    f"• Detections: "
                    f"{detection_count}"
                )

            else:

                status.text(
                    f"Processing frame "
                    f"{frame_number} "
                    f"• Detections: "
                    f"{detection_count}"
                )

    finally:

        cap.release()
        out.release()

    progress.empty()
    status.empty()

    return frame_number, detection_count


# ============================================================
# MAIN APP
# ============================================================

if uploaded_video is not None:

    st.subheader("🎬 Original Video")

    st.video(uploaded_video)

    if st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    ):

        input_path = None
        output_path = None

        try:

            # Save uploaded video
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as input_file:

                input_file.write(
                    uploaded_video.getbuffer()
                )

                input_path = input_file.name


            # Create output file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as output_file:

                output_path = output_file.name


            with st.spinner(
                "AI is analyzing the video frame-by-frame..."
            ):

                frames_processed, total_detections = (
                    process_video(
                        input_path,
                        output_path,
                        confidence,
                        image_size
                    )
                )


            st.success(
                f"✅ Analysis complete — "
                f"{frames_processed:,} frames processed."
            )


            # =================================================
            # DETECTION SUMMARY
            # =================================================

            st.subheader("📊 Detection Summary")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Frames Processed",
                    f"{frames_processed:,}"
                )

            with col2:

                st.metric(
                    "Total Detections",
                    f"{total_detections:,}"
                )


            # =================================================
            # DEMO EVENT
            # =================================================

            incident = random.choice(
                INCIDENT_EXAMPLES
            )

            st.subheader("📡 Detection Event")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Event Type",
                    incident["eventType"]
                )

            with col2:

                st.metric(
                    "Confidence",
                    f'{incident["detection"]["confidence"] * 100:.0f}%'
                )

            with col3:

                st.metric(
                    "Severity",
                    incident["detection"]["severity"]
                )


            # =================================================
            # LOCATION
            # =================================================

            st.write("### 📍 Location")

            location = incident["location"]

            st.write(
                f'**Address:** '
                f'{location["address"]}'
            )

            st.write(
                f'**Coordinates:** '
                f'{location["latitude"]}, '
                f'{location["longitude"]}'
            )


            # =================================================
            # BUS / CAMERA
            # =================================================

            st.write(
                "### 🚌 Vehicle / Camera"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f'**Bus ID:** '
                    f'{incident["busId"]}'
                )

            with col2:

                st.write(
                    f'**Camera:** '
                    f'{incident["cameraId"]}'
                )


            # =================================================
            # MODEL INFO
            # =================================================

            st.write(
                "### 🤖 Model Information"
            )

            st.write(
                f'**Model:** '
                f'{incident["model"]["name"]}  \n'
                f'**Version:** '
                f'{incident["model"]["version"]}'
            )


            # =================================================
            # JSON
            # =================================================

            with st.expander(
                "View Event JSON"
            ):

                st.json(incident)


            # =================================================
            # OUTPUT VIDEO
            # =================================================

            st.subheader(
                "🎥 Annotated Output Video"
            )

            st.video(output_path)


            # Download
            with open(
                output_path,
                "rb"
            ) as f:

                video_bytes = f.read()


            st.download_button(
                "⬇️ Download Annotated Video",
                data=video_bytes,
                file_name="road_damage_detected.mp4",
                mime="video/mp4",
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"❌ Error while processing "
                f"video: {e}"
            )

        finally:

            if (
                input_path
                and os.path.exists(input_path)
            ):

                os.remove(input_path)
