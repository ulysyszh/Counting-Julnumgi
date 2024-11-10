import streamlit as st
import warnings
import torch
from utils.jump_counter import JumpCounter
from utils.font_manager import download_font
import tempfile
import cv2
import time
from PIL import Image
import numpy as np
import os

# Suppress torch warnings
torch.classes.__module__ = "torch.classes"

# Create necessary directories
os.makedirs("assets", exist_ok=True)
os.makedirs("results", exist_ok=True)


def main():
    st.title("줄넘기 카운터")

    # Download font if not exists
    download_font()

    # Add input source selection
    input_source = st.radio("입력 소스 선택", ["웹캠", "비디오 파일"])

    if input_source == "비디오 파일":
        # Existing file upload logic
        uploaded_file = st.file_uploader(
            "비디오 파일을 선택하세요", type=["mp4", "avi"]
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            st.stop()
    else:  # Webcam
        video_path = 0  # Use default webcam (0)

    # Initialize counter
    counter = JumpCounter()

    # Process video/webcam button
    button_text = "웹캠 시작" if input_source == "웹캠" else "동영상 분석 시작"
    if st.button(button_text):
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Fix for webcam fps if not detected
            fps = 30

        # Create output video writer
        output_path = f"results/output_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Add stop button for webcam
        stop_button = st.empty()
        stop_webcam = False
        if input_source == "웹캠":
            stop_webcam = stop_button.button("웹캠 중지")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if stop_webcam:
                    break

                # Process frame
                results = counter.model(frame, verbose=False)

                if len(results) > 0 and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints.data[0].cpu().numpy()

                    # Add error handling for jump detection
                    try:
                        counter.detect_jumps(keypoints)
                    except RuntimeWarning:
                        continue

                    # Draw pose estimation
                    annotated_frame = results[0].plot()

                    # Add counter overlay
                    overlay = annotated_frame.copy()
                    counter.draw_counters(overlay)

                    # Convert BGR to RGB for streamlit
                    rgb_frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                    # Display frame
                    video_placeholder.image(
                        rgb_frame, channels="RGB", use_container_width=True
                    )

                # Update metrics
                # with metrics_placeholder.container():
                #     col1, col2, col3, col4 = st.columns(4)
                #     with col1:
                #         st.metric("기본뛰기", counter.normal_jump_counter)
                #     with col2:
                #         st.metric("번갈아뛰기", counter.knee_counter)
                #     with col3:
                #         st.metric("옆흔들어뛰기", counter.jumping_jack_counter)
                #     with col4:
                #         st.metric("엇걸어풀어뛰기", counter.criss_cross_counter * 2)

                # Write frame to output video
                # Write frame to output video only for file input
                if input_source == "비디오 파일":
                    out.write(overlay)

        finally:
            # Ensure proper cleanup
            cap.release()
            out.release()

            # Show success message and video playback
            st.success("처리 완료!")

            try:
                # Convert AVI to MP4 using cv2
                mp4_output = output_path.replace(".avi", ".mp4")
                cap = cv2.VideoCapture(output_path)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    mp4_output, fourcc, fps, (frame_width, frame_height)
                )

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                cap.release()
                out.release()

                # Display the video
                with open(mp4_output, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)

                # Clean up AVI file
                os.remove(output_path)

            except Exception as e:
                st.error(f"비디오 변환 중 오류 발생: {str(e)}")
                # Provide download link for AVI file as fallback
                st.markdown(f"[결과 비디오 다운로드]({output_path})")


if __name__ == "__main__":
    main()
