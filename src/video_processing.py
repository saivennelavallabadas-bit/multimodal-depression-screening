
import cv2
import numpy as np


def extract_video_features(video_path: str, sample_every_seconds: float = 1.0) -> dict:
    if sample_every_seconds <= 0:
        raise ValueError("Sampling interval must be positive.")

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Could not open the video file.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("The video has invalid frame metadata.")

    duration = frame_count / fps
    interval = max(1, int(round(fps * sample_every_seconds)))

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        capture.release()
        raise RuntimeError("Could not load the face detector.")

    sampled_frames = 0
    detected_faces = 0
    frames_with_faces = 0
    face_sizes = []
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % interval == 0:
                sampled_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )

                if len(faces) > 0:
                    frames_with_faces += 1

                detected_faces += len(faces)

                for x, y, width, height in faces:
                    frame_area = frame.shape[0] * frame.shape[1]
                    face_sizes.append(
                        (width * height) / frame_area
                    )

            frame_index += 1

    finally:
        capture.release()

    return {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": frame_count,
        "sampled_frames": sampled_frames,
        "frames_with_detected_faces": frames_with_faces,
        "total_face_detections": int(detected_faces),
        "mean_face_area_ratio": (
            round(float(np.mean(face_sizes)), 4)
            if face_sizes
            else None
        ),
    }
