"""Loading and validating the YAML configuration file."""
import copy
import yaml

DEFAULTS = {
    "project": {
        "name": "behavkit_project",
        "fps": 30,
        "random_state": 42,
    },
    "paths": {
        "behavsoft_dir": "behavsoft_ethograms",
        "dlc_dir": "dlc_outputs",
        "raw_video_dir": "raw_videos",
        "output_dir": "outputs",
        "annotated_video_dir": "outputs/videos",
        "holdout_split_path": "outputs/holdout_video_ids.csv",
        "model_report_path": "outputs/model_classification_report.csv",
        "confusion_matrix_path": "outputs/model_confusion_matrix.csv",
        "best_params_path": "outputs/model_best_params.json",
        "sequences_path": "outputs/behavior_sequences.csv",
        "threshold_metrics_path": "outputs/threshold_cv_metrics.csv",
        "activity_data_path": "outputs/activity_by_frame.csv",
        "bout_metrics_path": "outputs/bout_metrics.csv",
        "bout_details_path": "outputs/bout_details.csv",
        "xls_extension": ".xls",         # use ".xlsx" if your scoring sheets are in that format
        "h5_extension": ".h5",
    },
    "preprocessing": {
        "likelihood_threshold": None,   # e.g. 0.9 to enable interpolation of low-confidence points
        "interpolate_limit": 10,
        "label_map": None,              # e.g. {"Swim_raw": "Swim"}
        "exclude_label": "other",
    },
    "features": {
        "velocity_bodyparts": ["nose", "head", "body", "tail_base"],
        "distance_pairs": [],           # e.g. [["nose", "tail_base"]]
        "angle_triplets": [],           # e.g. [["nose", "body", "tail_base"]]
        "polygon_specs": [],            # e.g. [{"name": "torso_area", "vertices": [...], "anchor_pair": [...]}]
        "rolling_windows": [5, 15, 30],
    },
    "modeling": {
        "enabled": True,
        "base_feature_columns": None,   # null = derived automatically from velocity_bodyparts
        "fuse_labels": None,            # e.g. {"Freezing": "Immobility", "Floating": "Immobility"}
        "exclude_labels": [],           # e.g. ["Dive"] -- classes too rare in videos to model
        "param_grid": {
            "max_depth": [10, None],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 5],
        },
        "n_estimators_search": 100,
        "n_estimators_final": 500,
        "use_balanced_rf": False,
        "class_weight": "balanced_subsample",
        "cv_splits": 5,
        "scoring": "f1_macro",
    },
    "sequences": {
        "enabled": True,
        "behavior_column": "y_target_behavior",
    },
    "threshold_detection": {
        "enabled": False,
        "activity_columns": [],         # e.g. ["vel_nose_mean_15", "vel_head_mean_15"]
        "positive_labels": [],          # e.g. ["Freezing", "Floating"]
        "lower_activity_is_positive": True,
        "min_duration_frames": 30,
        "cv_splits": 5,
        "export_raw_activity_data": True,   # per-frame activity+label CSV, for external distribution/ROC analysis
    },
    "video_annotation": {
        "enabled": False,
        "mode": "classification",       # "classification" or "threshold"
        "example_video_id": None,       # null = use the first available video
        "filename_template": "{video_id}.mp4",  # how video_id maps to a file in raw_video_dir
        "class_colors": None,           # e.g. {"Immobility": [0, 0, 220]} (BGR)
    },
}


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base` (dicts merge key by key; other types replace)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path):
    """
    Load a YAML config file and merge it on top of `DEFAULTS`, so users only
    need to specify the parameters they actually want to change.
    """
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    return _deep_merge(DEFAULTS, user_config)
