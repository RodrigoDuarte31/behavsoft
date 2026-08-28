"""Reading DeepLabCut outputs (.h5) and manual scoring spreadsheets (Behavsoft/.xls)."""
import os
import pandas as pd
import numpy as np

from .utils import to_seconds, logger


def format_h5_file(h5_path, likelihood_threshold=None, interpolate_limit=10):
    """
    Read a DLC output (.h5, MultiIndex scorer/bodypart/coord columns) and
    return a flat DataFrame, with columns '{bodypart}_x', '{bodypart}_y',
    '{bodypart}_likelihood'.

    If `likelihood_threshold` is given, points below the threshold are set
    to NaN and linearly interpolated (capped at `interpolate_limit`
    consecutive frames, so a long occlusion doesn't get stretched into an
    artificial straight line).
    """
    df_dlc = pd.read_hdf(h5_path)
    df_dlc.columns = [f"{part}_{coord}" for _scorer, part, coord in df_dlc.columns]

    if likelihood_threshold is not None:
        bodyparts = sorted({c.rsplit('_', 1)[0] for c in df_dlc.columns if c.endswith('_likelihood')})
        for bp in bodyparts:
            low_confidence_mask = df_dlc[f'{bp}_likelihood'] < likelihood_threshold
            for coord in ('x', 'y'):
                col = f'{bp}_{coord}'
                df_dlc.loc[low_confidence_mask, col] = np.nan
                df_dlc[col] = df_dlc[col].interpolate(limit=interpolate_limit)

    return df_dlc


def discover_video_pairs(behavsoft_dir, dlc_dir, xls_ext=".xls", h5_ext=".h5", match_fn=None):
    """
    Find (scoring spreadsheet, DLC output) pairs across the two folders.

    `match_fn(name_without_extension, list_of_h5_files) -> matching h5 filename or None`
    controls how a .xls name is associated with its .h5 counterpart. If not
    given, defaults to substring matching (the .xls name must appear inside
    the .h5 filename) -- adjust for your own file naming convention.
    """
    xls_files = sorted(f for f in os.listdir(behavsoft_dir) if f.endswith(xls_ext))
    h5_files = sorted(f for f in os.listdir(dlc_dir) if f.endswith(h5_ext))

    if match_fn is None:
        def match_fn(name, h5_candidates):
            found = [f for f in h5_candidates if name in f]
            return found[0] if found else None

    pairs = []
    for xls in xls_files:
        name = xls.replace(xls_ext, "")
        h5_match = match_fn(name, h5_files)
        if h5_match is None:
            logger.warning(f"No matching .h5 found for {xls}.")
            continue
        pairs.append({
            "video_id": name,
            "xls_path": os.path.join(behavsoft_dir, xls),
            "h5_path": os.path.join(dlc_dir, h5_match),
        })

    return pairs


def load_ethogram(xls_path, label_map=None, usecols=(0, 1, 2, 3),
                   column_names=("behavior", "start", "end", "duration")):
    """
    Read a manual scoring spreadsheet (behavior, start, end, duration) and
    return the timestamps already converted to seconds, relative to the
    start of the first event.

    `label_map` (optional): dict to rename labels, e.g.
    {'Swim': 'Swimming', 'Freeze': 'Freezing'}.
    """
    df_behav = pd.read_excel(xls_path, usecols=list(usecols))
    df_behav.columns = list(column_names)

    if label_map:
        df_behav['behavior'] = df_behav['behavior'].replace(label_map)

    df_behav['start_abs'] = df_behav['start'].apply(lambda t: to_seconds(t, context=xls_path))
    df_behav['end_abs'] = df_behav['end'].apply(lambda t: to_seconds(t, context=xls_path))

    n_bad = df_behav['start_abs'].isna().sum() + df_behav['end_abs'].isna().sum()
    if n_bad > 0:
        logger.warning(f"{xls_path}: {n_bad} timestamp(s) failed to parse (see warnings above).")

    t0 = df_behav['start_abs'].min()
    df_behav['start_rel'] = df_behav['start_abs'] - t0
    df_behav['end_rel'] = df_behav['end_abs'] - t0

    return df_behav


def join_features_with_ethogram(df_feat, df_behav, fps, default_label="other"):
    """
    Assign, to each frame of `df_feat` (must have a sequential frame index,
    0..N-1), the behavior label from `df_behav` (columns 'behavior',
    'start_rel', 'end_rel', produced by `load_ethogram`). Frames outside any
    annotated interval get `default_label`.
    """
    df_feat = df_feat.copy()
    df_feat['time_s'] = df_feat.index / fps
    df_feat['y_target_behavior'] = default_label

    for _, row in df_behav.iterrows():
        start, end = row['start_rel'], row['end_rel']
        if pd.isna(start) or pd.isna(end):
            continue

        if start == end:
            mask = np.floor(df_feat['time_s']) == start
        else:
            mask = (df_feat['time_s'] >= start) & (df_feat['time_s'] < end)

        df_feat.loc[mask, 'y_target_behavior'] = row['behavior']

    return df_feat
