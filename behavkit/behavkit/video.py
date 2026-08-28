"""Generating annotated videos (multiclass classification or binary detection overlay)."""
import os
import cv2

from .utils import logger
from .threshold import apply_minimum_duration


def annotate_video_classification(video_path, predictions_per_frame, output_path,
                                   class_colors=None, default_color=(200, 200, 200)):
    """
    Overlay, on every frame of the original video, the label predicted by a
    multiclass classifier (e.g. Random Forest).

    `predictions_per_frame`: list/array of labels (str), one per frame, in
    the same order/frame rate as the video.
    `class_colors`: dict {label: (B, G, R)}. Labels without a defined color
    use `default_color`.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return

    class_colors = class_colors or {}

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i < len(predictions_per_frame):
            label = predictions_per_frame[frame_i]
            color = class_colors.get(label, default_color)
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 6)
            cv2.putText(frame, str(label), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_i += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {output_path}")


def annotate_video_threshold(video_path, activity_per_frame, threshold, output_path,
                              min_duration_frames=0, lower_activity_is_positive=True,
                              positive_color=(0, 200, 0), negative_color=(0, 0, 220),
                              positive_label="POSITIVE", negative_label="NEGATIVE"):
    """
    Overlay, on every frame, whether the activity crossed the threshold
    (e.g. immobility detection via a velocity cutoff), already with the
    minimum-duration filter applied.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return

    if lower_activity_is_positive:
        predictions = (activity_per_frame <= threshold).astype(int)
    else:
        predictions = (activity_per_frame >= threshold).astype(int)

    if min_duration_frames > 0:
        predictions = apply_minimum_duration(predictions, min_duration_frames)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i < len(predictions):
            is_positive = predictions[frame_i] == 1
            color = positive_color if is_positive else negative_color
            label = positive_label if is_positive else negative_label
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 6)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_i += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {output_path}")
