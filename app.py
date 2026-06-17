import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess

from main import FootballAnalytics
from config import config


# Page settings
st.set_page_config(
    page_title="Football Match Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Football Match Analyzer")

# UI controls
tracking_method = st.selectbox("Tracking Method", ["ByteTrack", "BoTSORT"])
save_output = st.checkbox("Save tracked video", value=True)
generate_report = st.checkbox("Generate Detailed Report", value=True)

st.info(
    "Run the app using Streamlit in your environment. On Windows, use: python -m streamlit run app.py"
)

# Ensure stable state across reruns (e.g., clicking Download)
st.session_state.setdefault("analysis_done", False)
st.session_state.setdefault("last_video_path", None)
st.session_state.setdefault("last_output_video_path", None)
st.session_state.setdefault("last_df", None)


# Sidebar
with st.sidebar:
    st.header("Settings")
    input_method = st.radio("Video Input Method:", ["Upload Video", "Video URL"])


# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Source")
    video_file = None
    video_path = None

    if input_method == "Upload Video":
        video_file = st.file_uploader(
            "Choose match video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: mp4, avi, mov, mkv",
        )
        if video_file is not None:
            # Store upload in a temp mp4 file to maximize decoder compatibility
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            video_path = tfile.name
            st.video(video_path)
            st.success(f"Uploaded: {video_file.name}")

    elif input_method == "Video URL":
        video_url = st.text_input("Enter video URL", placeholder="https://example.com/match.mp4")
        if video_url:
            video_path = video_url
            st.video(video_path)

with col2:
    st.subheader("Video Information")
    if video_path:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps and fps > 0 else 0
            cap.release()

            minutes = int(duration // 60)
            seconds = int(duration % 60)

            st.metric("Duration", f"{minutes}:{seconds:02d}")
            st.metric("Resolution", f"{width} * {height}")
            st.metric("FPS", f"{fps}")
        except Exception:
            st.warning("Could not read video metadata.")
    else:
        st.info("Waiting for video to load ...")


# Analyze button
_, mid, _ = st.columns([1, 2, 1])
with mid:
    analyze_clicked = st.button(
        "Start Analysis", type="primary", disabled=video_path is None
    )


def _try_convert_avi_to_mp4(path: Path) -> Path:
    if path.suffix.lower() != ".avi":
        return path

    mp4_path = path.with_suffix(".mp4")
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        return mp4_path

    try:
        with st.spinner("Converting AVI to MP4..."):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(mp4_path),
                ],
                check=False,
                capture_output=True,
            )
        if mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    except Exception:
        pass

    return path


def _newest_video_in_dir(base: Path) -> Path | None:
    patterns = ["*.mp4", "*.avi", "*.mkv", "*.MOV", "*.MP4", "*.AVI"]
    if not base.exists():
        return None
    candidates = []
    for p in patterns:
        candidates.extend(list(base.rglob(p)))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


# Run pipeline only when Start Analysis is clicked
if analyze_clicked and video_path:
    # reset per-run outputs but keep stable UI containers
    st.session_state.analysis_done = False
    st.session_state.last_video_path = str(video_path)
    st.session_state.last_output_video_path = None
    st.session_state.last_df = None

    # OpenCV validation
    cap_check = cv2.VideoCapture(video_path)
    if not cap_check.isOpened():
        cap_check.release()
        st.error(
            "OpenCV cannot open the selected video. Check file format/codec or upload a playable video."
        )
        st.stop()
    cap_check.release()

    # YOLO tracking
    status = st.empty()
    status.text("Loading model")
    model = YOLO("yolo11n.pt")

    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / f"temp_output_{datetime.now().strftime('%Y%M%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    status.text("Analyzing video")

    results_stream = model.track(
        source=video_path,
        save=save_output,
        project=str(output_dir),
        name="tracked_match",
        conf=config.confidence_threshold,
        iou=config.iou_threshold,
        persist=True,
        stream=True,
        verbose=False,
        save_txt=True,
        save_crop=False,
    )

    cap_tmp = cv2.VideoCapture(video_path)
    total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap_tmp.release()

    progress_bar = st.progress(0)
    frame_counter = 0
    for _ in results_stream:
        frame_counter += 1
        if frame_counter % 10 == 0:
            progress_bar.progress(min(1.0, frame_counter / total_frames))

    progress_bar.progress(1.0)
    status.text("Analyze complete")

    # Offline analytics (creates output/reports/report.xlsx)
    analytics = FootballAnalytics(input_video=video_path)
    analytics.process_video()


    report_path = repo_root / "output" / "reports" / "report.xlsx"

    if report_path.exists():
        df = pd.read_excel(report_path, sheet_name="Players")
    else:
        # fallback: some runs write directly via config.output_dir / reports / report.xlsx
        alt_report_path = Path("./output/reports/report.xlsx")
        if alt_report_path.exists():
            df = pd.read_excel(alt_report_path, sheet_name="Players")
        else:
            df = pd.DataFrame()
            st.warning(f"report.xlsx not found at: {report_path}")

    # Find newest tracked video for playback
    tracked_video = None
    if save_output:
        tracked_video = _newest_video_in_dir(output_dir / "tracked_match")
        if tracked_video is None:
            tracked_video = _newest_video_in_dir(output_dir)
        if tracked_video is None:
            # fallback to stable output folder
            stable = repo_root / "output" / "tracked_videos" / "output.mp4"
            if stable.exists():
                tracked_video = stable

    if tracked_video is not None:
        tracked_video = _try_convert_avi_to_mp4(Path(tracked_video))

    st.session_state.last_output_video_path = str(tracked_video) if tracked_video else None
    st.session_state.last_df = df
    st.session_state.analysis_done = True


# Render results if analysis already done
if st.session_state.analysis_done:
    st.header("Analyze Results")

    tab1, tab2, tab3 = st.tabs(["Tracked video", "Heatmaps", "Player Analysis"])

    with tab1:
        st.subheader("Tracked video")
        if st.session_state.last_output_video_path:
            video_to_show = Path(st.session_state.last_output_video_path)

            # Download button (this triggers rerun but state keeps results visible)
            try:
                with open(video_to_show, "rb") as f:
                    st.download_button(
                        "📥 Download Video",
                        f,
                        file_name=video_to_show.name,
                        mime="video/mp4" if video_to_show.suffix.lower() == ".mp4" else None,
                    )
            except Exception:
                pass

            # Preview first frame
            try:
                cap_prev = cv2.VideoCapture(str(video_to_show))
                ret_prev, frame_prev = cap_prev.read()
                cap_prev.release()
                if ret_prev and frame_prev is not None:
                    st.image(
                        cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB),
                        caption="📸 Preview (first frame)",
                        use_container_width=True,
                    )
            except Exception:
                pass

        else:
            st.warning("Tracked video not found.")

    with tab2:
        st.subheader("player heatmap")
        # Placeholder heatmap until you export real grid heat data
        heat_data = np.zeros((50, 34), dtype=np.float32)
        fig, ax = plt.subplots(figsize=(10, 7))

        # Use nearest interpolation to avoid "triangles"/interpolation artifacts on grid data
        im = ax.imshow(
            heat_data,
            cmap="hot",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="auto",
        )

        ax.set_title("Player positioning heatmap")
        ax.set_xlabel("Pitch Width")
        ax.set_ylabel("Pitch Length")
        plt.colorbar(im, ax=ax, label="Intensity")
        st.pyplot(fig)

    with tab3:
        st.subheader("Indivivual Player analysis")
        df = st.session_state.last_df if st.session_state.last_df is not None else pd.DataFrame()

        if df.empty:
            st.warning("No player metrics to display.")
        else:
            st.dataframe(df, width="stretch")

            sub1, sub2 = st.tabs(["Visualization", "Visualization 2"])
            with sub1:
                fig1 = px.bar(
                    df,
                    x="Player_id",
                    y=["Max Speed", "Avg Speed"],
                    barmode="group",
                    title="Player Speed (Max vs Avg) [m/s]",
                )
                st.plotly_chart(fig1, use_container_width=True)

            with sub2:
                if "Totale Distance" in df.columns and "Avg Speed" in df.columns:
                    fig2 = px.scatter(
                        df,
                        x="Avg Speed",
                        y="Totale Distance",
                        color="Player_id",
                        size="Totale Distance",
                        title="Avg Speed vs Total Distance",
                    )
                    st.plotly_chart(fig2, use_container_width=True)