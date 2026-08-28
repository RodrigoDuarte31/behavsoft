"""
behavkit -- a reusable pipeline for DeepLabCut + Behavsoft data.

Scope: preprocessing, feature engineering, threshold-based binary detection
(e.g. immobility), and Random Forest classification -- with CSV exports for
statistical analysis and plotting in external tools. Also generates
annotated example videos.

Modules:
    config      YAML configuration loading
    run         main pipeline entry point (run_pipeline)
    io          reading .h5 (DLC) and .xls (manual scoring) files
    features    kinematic feature extraction (generic w.r.t. bodyparts)
    pipeline    batch preprocessing orchestration
    modeling    multiclass classification (Random Forest) with video-grouped CV
    sequences   ordered behavior-event export (for external Markov/sequence analysis)
    threshold   binary threshold detection, minimum-duration filter, bout metrics
    video       annotated videos (classification or threshold detection overlay)

Quick start (config-driven, recommended):
    import behavkit as bk

    results = bk.run_pipeline("config.yaml")
    # results["df_train"], results["modeling"]["model"], results["threshold_detection"]["threshold"], ...
    # CSVs are written to the paths defined in config.yaml -- open them in
    # your stats/plotting tool of choice.

Quick start (manual, function by function):
    import behavkit as bk

    bk.setup_logging()
    pairs = bk.discover_video_pairs("behavsoft/", "dlc_outputs/")
    df_train, df_full = bk.preprocess_batch(
        pairs,
        feature_kwargs=dict(velocity_bodyparts=['nose', 'head', 'body', 'tail_base']),
        fps=30,
    )
"""
from .config import load_config, DEFAULTS
from .run import run_pipeline
from .utils import setup_logging, to_seconds
from .io import format_h5_file, discover_video_pairs, load_ethogram, join_features_with_ethogram
from .features import extract_kinematics_features, calc_velocity, calc_distance, calc_angle
from .pipeline import preprocess_batch
from .modeling import (
    build_feature_list, get_or_create_group_holdout, grid_search_group_cv,
    train_final_model, evaluate_model,
)
from .sequences import extract_behavior_events, export_behavior_sequences
from .threshold import (
    get_bouts, apply_minimum_duration, apply_minimum_duration_per_video,
    select_threshold_group_cv, compute_bout_metrics, batch_bout_metrics,
    bouts_dataframe_for_video, batch_bouts_dataframe,
)
from .video import annotate_video_classification, annotate_video_threshold

__version__ = "0.3.0"

__all__ = [
    "load_config", "DEFAULTS", "run_pipeline",
    "setup_logging", "to_seconds",
    "format_h5_file", "discover_video_pairs", "load_ethogram", "join_features_with_ethogram",
    "extract_kinematics_features", "calc_velocity", "calc_distance", "calc_angle",
    "preprocess_batch",
    "build_feature_list", "get_or_create_group_holdout", "grid_search_group_cv",
    "train_final_model", "evaluate_model",
    "extract_behavior_events", "export_behavior_sequences",
    "get_bouts", "apply_minimum_duration", "apply_minimum_duration_per_video",
    "select_threshold_group_cv", "compute_bout_metrics", "batch_bout_metrics",
    "bouts_dataframe_for_video", "batch_bouts_dataframe",
    "annotate_video_classification", "annotate_video_threshold",
]
