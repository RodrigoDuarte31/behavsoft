"""Orchestrates batch preprocessing: multiple videos -> a single dataset."""
import pandas as pd

from .io import format_h5_file, load_ethogram, join_features_with_ethogram
from .features import extract_kinematics_features
from .utils import logger


def preprocess_batch(video_pairs, feature_kwargs, fps, label_map=None,
                      likelihood_threshold=None, exclude_label="other"):
    """
    Process a list of videos (output of `discover_video_pairs`, or a
    manually built list of {'video_id', 'xls_path', 'h5_path'} dicts) and
    concatenate everything into two DataFrames:

    - df_train: only annotated frames (excludes `exclude_label`) -- for
      training/evaluation.
    - df_full: ALL frames from all videos, including unannotated ones --
      needed to generate continuous predictions (annotated videos, per-video
      bout metrics over the whole recording).

    `feature_kwargs`: dict forwarded to `extract_kinematics_features`
    (velocity_bodyparts, distance_pairs, angle_triplets, etc.).
    """
    dfs = []

    for pair in video_pairs:
        video_id = pair['video_id']
        print(f"Processing video: {video_id}")
        try:
            df_raw = format_h5_file(pair['h5_path'], likelihood_threshold=likelihood_threshold)
            df_feat = extract_kinematics_features(df_raw, **feature_kwargs)

            df_behav = load_ethogram(pair['xls_path'], label_map=label_map)
            df_final = join_features_with_ethogram(df_feat, df_behav, fps=fps, default_label=exclude_label)

            df_final['video_id'] = video_id
            dfs.append(df_final)
        except Exception as e:
            logger.error(f"Could not process video {video_id}: {e}")

    if not dfs:
        raise RuntimeError("No video was processed successfully -- check paths and file formats.")

    df_full = pd.concat(dfs, ignore_index=True)
    df_train = df_full[df_full['y_target_behavior'] != exclude_label].copy().dropna()

    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"Annotated frames: {len(df_train)} | Total frames: {len(df_full)} | Videos: {df_train['video_id'].nunique()}")
    print("\nClass distribution:")
    print(df_train['y_target_behavior'].value_counts())

    return df_train, df_full
