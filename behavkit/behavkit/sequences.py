"""
Export ordered behavior-event sequences, one row per event, for Markov-chain
(or any other sequence) analysis in an external tool (R, MATLAB, etc.).

Uses `df_full` (not `df_train`) by default, so 'other'/unannotated frames
appear as their own event -- this preserves true temporal contiguity, which
matters for interpreting adjacency correctly. Filtering out 'other' events,
if desired, is left to the external analysis tool.
"""
import pandas as pd


def extract_behavior_events(df, behavior_col='y_target_behavior', video_col='video_id', time_col='time_s'):
    """
    Collapse consecutive identical behavior labels, per video, into discrete
    ordered events. Returns a tidy DataFrame with one row per event:
    video_id, event_order (0-indexed, per video), behavior, start_s, end_s,
    duration_s.
    """
    rows = []

    for video_id, group in df.sort_values(time_col).groupby(video_col, sort=False):
        times = group[time_col].to_numpy()
        behaviors = group[behavior_col].to_numpy()

        event_order = 0
        current_behavior = behaviors[0]
        start_time = times[0]

        for i in range(1, len(behaviors)):
            if behaviors[i] != current_behavior:
                rows.append({
                    'video_id': video_id, 'event_order': event_order,
                    'behavior': current_behavior, 'start_s': start_time,
                    'end_s': times[i - 1], 'duration_s': times[i - 1] - start_time,
                })
                event_order += 1
                current_behavior = behaviors[i]
                start_time = times[i]

        rows.append({
            'video_id': video_id, 'event_order': event_order,
            'behavior': current_behavior, 'start_s': start_time,
            'end_s': times[-1], 'duration_s': times[-1] - start_time,
        })

    return pd.DataFrame(rows)


def export_behavior_sequences(df, output_path, behavior_col='y_target_behavior',
                               video_col='video_id', time_col='time_s'):
    """Compute the event sequence (see `extract_behavior_events`) and write it to `output_path` as CSV."""
    df_events = extract_behavior_events(df, behavior_col=behavior_col, video_col=video_col, time_col=time_col)
    df_events.to_csv(output_path, index=False)
    return df_events
