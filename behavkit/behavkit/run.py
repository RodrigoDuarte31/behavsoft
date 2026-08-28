"""Main pipeline entry point -- driven entirely by a YAML config file."""
import os
import json
import numpy as np
import pandas as pd

from .config import load_config
from .utils import setup_logging, logger
from .io import discover_video_pairs
from .pipeline import preprocess_batch
from .modeling import (
    build_feature_list, get_or_create_group_holdout, grid_search_group_cv,
    train_final_model, evaluate_model,
)
from .sequences import export_behavior_sequences
from .threshold import select_threshold_group_cv, batch_bout_metrics, batch_bouts_dataframe
from .video import annotate_video_classification, annotate_video_threshold


def _run_modeling(cfg, df_train):
    logger.info("=== Modeling ===")
    m_cfg = cfg["modeling"]

    df_model = df_train.copy()
    if m_cfg["exclude_labels"]:
        df_model = df_model[~df_model["y_target_behavior"].isin(m_cfg["exclude_labels"])]

    base_columns = m_cfg["base_feature_columns"]
    if base_columns is None:
        base_columns = [f'vel_{bp}' for bp in cfg["features"]["velocity_bodyparts"]]

    feature_columns = build_feature_list(base_columns, windows=cfg["features"]["rolling_windows"])

    X = df_model[feature_columns].astype(np.float32)
    y = df_model["y_target_behavior"]
    groups = df_model["video_id"]

    split_path = cfg["paths"]["holdout_split_path"]
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)

    train_mask, test_mask = get_or_create_group_holdout(
        X, y, groups, split_path=split_path,
        n_splits=m_cfg["cv_splits"], random_state=cfg["project"]["random_state"],
    )
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]

    search = grid_search_group_cv(
        X_train, y_train, groups_train, m_cfg["param_grid"],
        n_estimators_search=m_cfg["n_estimators_search"], n_splits=m_cfg["cv_splits"],
        use_balanced_rf=m_cfg["use_balanced_rf"], class_weight=m_cfg["class_weight"],
        scoring=m_cfg["scoring"], random_state=cfg["project"]["random_state"],
    )
    logger.info(f"Best hyperparameters: {search.best_params_}")

    model = train_final_model(
        search.best_params_, X_train, y_train,
        n_estimators=m_cfg["n_estimators_final"], use_balanced_rf=m_cfg["use_balanced_rf"],
        class_weight=m_cfg["class_weight"], random_state=cfg["project"]["random_state"],
    )
    report_text, report_df, cm, labels = evaluate_model(model, X_test, y_test)
    print(report_text)

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    report_df.to_csv(cfg["paths"]["model_report_path"])
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cfg["paths"]["confusion_matrix_path"])
    with open(cfg["paths"]["best_params_path"], "w") as f:
        json.dump(search.best_params_, f, indent=2, default=str)
    logger.info(f"Classification report saved to {cfg['paths']['model_report_path']}")
    logger.info(f"Confusion matrix saved to {cfg['paths']['confusion_matrix_path']}")

    return {
        "model": model, "best_params": search.best_params_,
        "report_text": report_text, "report_df": report_df,
        "confusion_matrix": cm, "labels": labels, "feature_columns": feature_columns,
    }


def _run_sequences(cfg, df_full):
    logger.info("=== Behavior sequence export ===")
    s_cfg = cfg["sequences"]

    os.makedirs(os.path.dirname(cfg["paths"]["sequences_path"]) or ".", exist_ok=True)
    df_events = export_behavior_sequences(
        df_full, cfg["paths"]["sequences_path"], behavior_col=s_cfg["behavior_column"],
    )
    logger.info(f"Behavior sequences ({len(df_events)} events) saved to {cfg['paths']['sequences_path']}")
    return {"sequences": df_events}


def _run_threshold_detection(cfg, df_train, df_full):
    logger.info("=== Threshold detection ===")
    t_cfg = cfg["threshold_detection"]

    if not t_cfg["activity_columns"]:
        raise ValueError("threshold_detection.enabled is true but activity_columns is empty in the config.")
    if not t_cfg["positive_labels"]:
        raise ValueError("threshold_detection.enabled is true but positive_labels is empty in the config.")

    df_train = df_train.copy()
    df_full = df_full.copy()
    df_train["overall_activity"] = df_train[t_cfg["activity_columns"]].mean(axis=1)
    df_full["overall_activity"] = df_full[t_cfg["activity_columns"]].mean(axis=1)

    activity = df_train["overall_activity"].to_numpy()
    y_binary = df_train["y_target_behavior"].isin(t_cfg["positive_labels"]).astype(int).to_numpy()
    groups = df_train["video_id"].to_numpy()

    threshold, fold_metrics = select_threshold_group_cv(
        activity, y_binary, groups, n_splits=t_cfg["cv_splits"],
        min_duration_frames=t_cfg["min_duration_frames"],
        random_state=cfg["project"]["random_state"],
        lower_activity_is_positive=t_cfg["lower_activity_is_positive"],
    )
    logger.info(f"Selected threshold (median across folds): {threshold:.4f}")
    print(fold_metrics)

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    fold_metrics.to_csv(cfg["paths"]["threshold_metrics_path"], index=False)
    logger.info(f"Threshold CV metrics saved to {cfg['paths']['threshold_metrics_path']}")

    if t_cfg["export_raw_activity_data"]:
        activity_cols = ["video_id", "time_s", "overall_activity", "y_target_behavior"]
        df_train["is_positive"] = y_binary
        df_train[activity_cols + ["is_positive"]].to_csv(cfg["paths"]["activity_data_path"], index=False)
        logger.info(f"Raw activity data saved to {cfg['paths']['activity_data_path']}")

    bout_metrics = batch_bout_metrics(
        df_full, "overall_activity", threshold, t_cfg["min_duration_frames"], cfg["project"]["fps"],
        lower_activity_is_positive=t_cfg["lower_activity_is_positive"],
    )
    bout_metrics.to_csv(cfg["paths"]["bout_metrics_path"], index=False)
    logger.info(f"Bout metrics saved to {cfg['paths']['bout_metrics_path']}")

    bout_details = batch_bouts_dataframe(
        df_full, "overall_activity", threshold, t_cfg["min_duration_frames"], cfg["project"]["fps"],
        target_labels=t_cfg["positive_labels"],
        lower_activity_is_positive=t_cfg["lower_activity_is_positive"],
    )
    bout_details.to_csv(cfg["paths"]["bout_details_path"], index=False)
    logger.info(f"Bout-level detail (annotation vs. prediction) saved to {cfg['paths']['bout_details_path']}")

    return {
        "threshold": threshold, "fold_metrics": fold_metrics,
        "bout_metrics": bout_metrics, "bout_details": bout_details,
        "df_full_with_activity": df_full,
    }


def _run_video_annotation(cfg, df_train, df_full, modeling_results, threshold_results):
    logger.info("=== Video annotation ===")
    v_cfg = cfg["video_annotation"]

    video_id = v_cfg["example_video_id"] or df_train["video_id"].unique()[0]
    video_path = os.path.join(cfg["paths"]["raw_video_dir"], v_cfg["filename_template"].format(video_id=video_id))
    output_path = os.path.join(cfg["paths"]["annotated_video_dir"], f"{video_id}_annotated.mp4")
    os.makedirs(cfg["paths"]["annotated_video_dir"], exist_ok=True)

    if v_cfg["mode"] == "classification":
        if modeling_results is None:
            raise ValueError("video_annotation.mode is 'classification' but modeling.enabled is false.")
        model = modeling_results["model"]
        feature_columns = modeling_results["feature_columns"]
        df_video = df_full[df_full["video_id"] == video_id].sort_values("time_s")
        predictions = model.predict(df_video[feature_columns].astype(np.float32))

        class_colors = None
        if v_cfg["class_colors"]:
            class_colors = {k: tuple(v) for k, v in v_cfg["class_colors"].items()}

        annotate_video_classification(video_path, predictions, output_path, class_colors=class_colors)

    elif v_cfg["mode"] == "threshold":
        if threshold_results is None:
            raise ValueError("video_annotation.mode is 'threshold' but threshold_detection.enabled is false.")
        df_full_activity = threshold_results["df_full_with_activity"]
        df_video = df_full_activity[df_full_activity["video_id"] == video_id].sort_values("time_s")
        activity = df_video["overall_activity"].to_numpy()

        annotate_video_threshold(
            video_path, activity, threshold_results["threshold"], output_path,
            min_duration_frames=cfg["threshold_detection"]["min_duration_frames"],
            lower_activity_is_positive=cfg["threshold_detection"]["lower_activity_is_positive"],
        )
    else:
        raise ValueError(f"Unknown video_annotation.mode: {v_cfg['mode']!r} (expected 'classification' or 'threshold')")


def run_pipeline(config_path):
    """
    Run the full behavkit pipeline end to end, driven entirely by a YAML
    config file (see config.example.yaml for every available parameter).

    Steps, each toggled independently in the config:
        1. Discover (scoring spreadsheet, DLC output) video pairs
        2. Batch preprocessing (feature extraction + ethogram join)
        3. Multiclass classification (Random Forest), if modeling.enabled
           -> classification report + confusion matrix saved as CSV
        4. Behavior sequence export, if sequences.enabled
           -> ordered per-video events saved as CSV, for Markov/sequence
              analysis in an external tool
        5. Binary threshold detection + bout metrics, if threshold_detection.enabled
           -> threshold CV metrics, raw activity data, aggregated bout
              metrics, and bout-level detail all saved as CSV
        6. Annotated example video, if video_annotation.enabled

    Returns a dict with every intermediate artifact (dataframes, model,
    thresholds, metrics DataFrames) for further interactive use.
    """
    cfg = load_config(config_path)
    setup_logging()

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    logger.info("=== Discovering video pairs ===")
    video_pairs = discover_video_pairs(
        cfg["paths"]["behavsoft_dir"], cfg["paths"]["dlc_dir"],
        xls_ext=cfg["paths"]["xls_extension"], h5_ext=cfg["paths"]["h5_extension"],
    )
    logger.info(f"Found {len(video_pairs)} video pair(s).")

    logger.info("=== Preprocessing ===")
    feature_kwargs = dict(
        velocity_bodyparts=cfg["features"]["velocity_bodyparts"],
        distance_pairs=[tuple(p) for p in cfg["features"]["distance_pairs"]],
        angle_triplets=[tuple(t) for t in cfg["features"]["angle_triplets"]],
        polygon_specs=cfg["features"]["polygon_specs"],
        rolling_windows=cfg["features"]["rolling_windows"],
    )
    df_train, df_full = preprocess_batch(
        video_pairs, feature_kwargs=feature_kwargs, fps=cfg["project"]["fps"],
        label_map=cfg["preprocessing"]["label_map"],
        likelihood_threshold=cfg["preprocessing"]["likelihood_threshold"],
        exclude_label=cfg["preprocessing"]["exclude_label"],
    )

    if cfg["modeling"]["fuse_labels"]:
        df_train["y_target_behavior"] = df_train["y_target_behavior"].replace(cfg["modeling"]["fuse_labels"])
        df_full["y_target_behavior"] = df_full["y_target_behavior"].replace(cfg["modeling"]["fuse_labels"])

    results = {"config": cfg, "df_train": df_train, "df_full": df_full}

    modeling_results = None
    if cfg["modeling"]["enabled"]:
        modeling_results = _run_modeling(cfg, df_train)
        results["modeling"] = modeling_results

    if cfg["sequences"]["enabled"]:
        results["sequences"] = _run_sequences(cfg, df_full)

    threshold_results = None
    if cfg["threshold_detection"]["enabled"]:
        threshold_results = _run_threshold_detection(cfg, df_train, df_full)
        results["threshold_detection"] = threshold_results

    if cfg["video_annotation"]["enabled"]:
        _run_video_annotation(cfg, df_train, df_full, modeling_results, threshold_results)

    logger.info("=== Pipeline complete ===")
    return results
