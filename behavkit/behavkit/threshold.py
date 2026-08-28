"""Binary activity-threshold detection -- CV-based selection, minimum-duration filter, bout metrics."""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, f1_score


def get_bouts(binary_predictions):
    """List of (start, end) for continuous runs of value 1 (end is exclusive)."""
    changes = np.diff(np.concatenate([[0], binary_predictions, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def apply_minimum_duration(binary_predictions, min_duration_frames):
    """
    Remove positive bouts shorter than `min_duration_frames` (they become 0
    again). Does not fill gaps -- only discards spikes too short to be
    behaviorally relevant (a common criterion in established behavior
    tools, e.g. EthoVision, ANY-maze).
    """
    predictions = binary_predictions.copy()
    for start, end in get_bouts(binary_predictions):
        if (end - start) < min_duration_frames:
            predictions[start:end] = 0
    return predictions


def apply_minimum_duration_per_video(binary_predictions, video_ids, min_duration_frames):
    """Same logic as above, applied video by video -- prevents a bout from leaking across concatenated videos."""
    predictions = binary_predictions.copy()
    video_ids = np.asarray(video_ids)
    for vid in pd.unique(video_ids):
        mask = video_ids == vid
        predictions[mask] = apply_minimum_duration(predictions[mask], min_duration_frames)
    return predictions


def select_threshold_group_cv(activity, y_binary, groups, n_splits=5,
                               min_duration_frames=0, random_state=42,
                               lower_activity_is_positive=True):
    """
    Select the optimal threshold (Youden's J = sensitivity + specificity - 1)
    with video-grouped cross-validation (`StratifiedGroupKFold`): in each
    fold, the cutoff is chosen only on the training videos and evaluated on
    the test videos (never seen during selection). Returns
    (final_threshold, DataFrame with per-fold metrics).

    `lower_activity_is_positive=True`: LOW activity indicates the positive
    class (e.g. immobility). Set to False if it's the other way around
    (e.g. detecting an activity spike).
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sign = -1 if lower_activity_is_positive else 1

    thresholds, metrics = [], []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(activity, y_binary, groups=groups)):
        X_tr, X_te = activity[train_idx], activity[test_idx]
        y_tr, y_te = y_binary[train_idx], y_binary[test_idx]
        groups_te = np.asarray(groups)[test_idx]

        fpr, tpr, thr = roc_curve(y_tr, sign * X_tr)
        best_threshold = sign * thr[np.argmax(tpr - fpr)]

        if lower_activity_is_positive:
            y_pred_te = (X_te <= best_threshold).astype(int)
        else:
            y_pred_te = (X_te >= best_threshold).astype(int)

        if min_duration_frames > 0:
            y_pred_te = apply_minimum_duration_per_video(y_pred_te, groups_te, min_duration_frames)

        sensitivity = (y_pred_te[y_te == 1] == 1).mean()
        specificity = (y_pred_te[y_te == 0] == 0).mean()

        thresholds.append(best_threshold)
        metrics.append({
            'fold': fold_i + 1, 'threshold': best_threshold,
            'sensitivity': sensitivity, 'specificity': specificity,
            'f1': f1_score(y_te, y_pred_te),
        })

    df_metrics = pd.DataFrame(metrics)
    final_threshold = float(np.median(thresholds))
    return final_threshold, df_metrics


def compute_bout_metrics(df_video, activity_col, threshold, min_duration_frames, fps,
                          time_col='time_s', lower_activity_is_positive=True, decimals=2):
    """
    Bout metrics for ONE video: frequency, total/mean duration, latency to
    the first episode (NaN if it never occurred -- made explicit in the
    table, instead of silently becoming 0).
    """
    df_video = df_video.sort_values(time_col)
    activity = df_video[activity_col].to_numpy()

    if lower_activity_is_positive:
        predictions = (activity <= threshold).astype(int)
    else:
        predictions = (activity >= threshold).astype(int)

    if min_duration_frames > 0:
        predictions = apply_minimum_duration(predictions, min_duration_frames)

    bouts = get_bouts(predictions)
    time_s = df_video[time_col].to_numpy()

    durations_s = [(end - start) / fps for start, end in bouts]
    return {
        'n_bouts': len(bouts),
        'total_duration_s': round(float(np.sum(durations_s)), decimals) if bouts else 0.0,
        'mean_duration_s': round(float(np.mean(durations_s)), decimals) if bouts else np.nan,
        'first_onset_latency_s': round(float(time_s[bouts[0][0]]), decimals) if bouts else np.nan,
        'video_duration_s': round(float(time_s[-1] - time_s[0]), decimals) if len(time_s) else np.nan,
    }


def batch_bout_metrics(df_full, activity_col, threshold, min_duration_frames, fps,
                        group_col='video_id', time_col='time_s', lower_activity_is_positive=True):
    """Applies `compute_bout_metrics` to every video in `df_full` and returns a consolidated DataFrame."""
    results = []
    for video_id, df_video in df_full.groupby(group_col, sort=False):
        m = compute_bout_metrics(df_video, activity_col, threshold, min_duration_frames, fps,
                                  time_col=time_col, lower_activity_is_positive=lower_activity_is_positive)
        m[group_col] = video_id
        results.append(m)

    columns = [group_col, 'n_bouts', 'total_duration_s', 'mean_duration_s',
               'first_onset_latency_s', 'video_duration_s']
    return pd.DataFrame(results)[columns]


def bouts_dataframe_for_video(df_video, activity_col, threshold, min_duration_frames, fps,
                               target_labels=None, behavior_col='y_target_behavior',
                               time_col='time_s', video_col='video_id', lower_activity_is_positive=True):
    """
    Bout-level detail for ONE video, long format: one row per bout, with a
    'source' column ('annotation' or 'prediction') -- everything needed to
    reconstruct an alignment raster (or any other bout-level analysis) in an
    external tool, without this package doing any plotting itself.

    If `target_labels` is given, annotation bouts are also extracted
    (frames where `behavior_col` is in `target_labels`) alongside the
    threshold-based prediction bouts. If `target_labels` is None, only
    prediction bouts are returned.
    """
    df_video = df_video.sort_values(time_col)
    time_s = df_video[time_col].to_numpy()
    video_id = df_video[video_col].iloc[0] if video_col in df_video.columns else None

    rows = []

    def _add_rows(binary_array, source):
        for start, end in get_bouts(binary_array):
            rows.append({
                'video_id': video_id, 'source': source,
                'start_s': float(time_s[start]), 'end_s': float(time_s[end - 1]) + (1 / fps),
                'duration_s': float(time_s[end - 1] - time_s[start]) + (1 / fps),
            })

    if target_labels is not None:
        y_true = df_video[behavior_col].isin(target_labels).astype(int).to_numpy()
        _add_rows(y_true, 'annotation')

    activity = df_video[activity_col].to_numpy()
    if lower_activity_is_positive:
        y_pred = (activity <= threshold).astype(int)
    else:
        y_pred = (activity >= threshold).astype(int)
    if min_duration_frames > 0:
        y_pred = apply_minimum_duration(y_pred, min_duration_frames)
    _add_rows(y_pred, 'prediction')

    return pd.DataFrame(rows, columns=['video_id', 'source', 'start_s', 'end_s', 'duration_s'])


def batch_bouts_dataframe(df_full, activity_col, threshold, min_duration_frames, fps,
                           target_labels=None, behavior_col='y_target_behavior',
                           time_col='time_s', group_col='video_id', lower_activity_is_positive=True):
    """Applies `bouts_dataframe_for_video` to every video in `df_full` and concatenates the results."""
    parts = []
    for _video_id, df_video in df_full.groupby(group_col, sort=False):
        parts.append(bouts_dataframe_for_video(
            df_video, activity_col, threshold, min_duration_frames, fps,
            target_labels=target_labels, behavior_col=behavior_col,
            time_col=time_col, video_col=group_col, lower_activity_is_positive=lower_activity_is_positive,
        ))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['video_id', 'source', 'start_s', 'end_s', 'duration_s'])
