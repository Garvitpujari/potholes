import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import gdown

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Road Damage Detection")
st.caption("YOLOv8 Small — Road Damage Detection")

MODEL_PATH = Path("YOLOv8_Small_RDD.pt")

# Google Drive file ID
DRIVE_FILE_ID = "1B0XF-Mnn62Wrv3G9IPeMaavkpO6ReAyl"


@st.cache_resource
def load_model():

    # Download model automatically if not already present
    if not MODEL_PATH.exists():

        with st.spinner("Downloading road damage model..."):

            gdown.download(
                id=DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False
            )

    return YOLO(str(MODEL_PATH))


# Load model
try:
    model = load_model()
    st.success("Road damage model loaded successfully.")

except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# Confidence threshold
confidence = st.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)


# Video upload
uploaded_video = st.file_uploader(
    "Upload a road video",
    type=["mp4", "avi", "mov", "mkv"]
)


def process_video(input_path, output_path, confidence):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    progress = st.progress(0)
    status = st.empty()

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO inference
        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

        # Draw detections
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        frame_number += 1

        if total_frames > 0:
            progress.progress(
                min(frame_number / total_frames, 1.0)
            )

        status.text(
            f"Processing frame {frame_number}"
            + (
                f" / {total_frames}"
                if total_frames > 0
                else ""
            )
        )

    cap.release()
    out.release()

    progress.empty()
    status.empty()


if uploaded_video is not None:

    st.subheader("Original Video")
    st.video(uploaded_video)

    if st.button(
        "🔍 Detect Road Damage",
        type="primary"
    ):

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(uploaded_video.read())
        input_file.close()

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()

        try:

            with st.spinner("Running road damage detection..."):

                process_video(
                    input_file.name,
                    output_path,
                    confidence
                )

            st.success("Detection complete!")

            st.subheader("Detected Road Damage")

            st.video(output_path)

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_bytes,
                file_name="road_damage_detected.mp4",
                mime="video/mp4"
            )

        except Exception as e:

            st.error(
                f"Error while processing video: {e}"
            )

        finally:

            if os.path.exists(input_file.name):
                os.remove(input_file.name)
