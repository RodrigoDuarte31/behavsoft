"""
Kinematic feature extraction from DLC coordinates (x, y).

All functions are generic with respect to bodypart names -- they work both
for the scheme used in the forced swim test (nose/head/body/tail_base/...)
and for any other scheme (e.g. open field), as long as the
'{bodypart}_x'/'{bodypart}_y' columns exist in the DataFrame.
"""
import numpy as np
import pandas as pd


def calc_velocity(df, bodypart):
    """Euclidean distance between consecutive frames (magnitude, no direction sign)."""
    dx = df[f'{bodypart}_x'].diff()
    dy = df[f'{bodypart}_y'].diff()
    return np.sqrt(dx**2 + dy**2)


def calc_distance(df, bp_a, bp_b):
    """Euclidean distance between two bodyparts, within the same frame."""
    return np.sqrt(
        (df[f'{bp_a}_x'] - df[f'{bp_b}_x'])**2 +
        (df[f'{bp_a}_y'] - df[f'{bp_b}_y'])**2
    )


def calc_angle(df, bp_a, bp_vertex, bp_c):
    """Angle (degrees) at `bp_vertex`, formed by the vectors vertex->a and vertex->c."""
    ax = df[f'{bp_a}_x'] - df[f'{bp_vertex}_x']
    ay = df[f'{bp_a}_y'] - df[f'{bp_vertex}_y']
    cx = df[f'{bp_c}_x'] - df[f'{bp_vertex}_x']
    cy = df[f'{bp_c}_y'] - df[f'{bp_vertex}_y']

    norm_a = np.sqrt(ax**2 + ay**2)
    norm_c = np.sqrt(cx**2 + cy**2)
    cos_angle = (ax * cx + ay * cy) / (norm_a * norm_c + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _triangle_area(x1, y1, x2, y2, x3, y3):
    """Area of a triangle given by 3 vertices (Shoelace formula for 3 points)."""
    return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def calc_polygon_area(df, vertices, anchor_pair):
    """
    Area of an N-vertex polygon (list of bodyparts, in any order),
    triangulated from a fixed diagonal `anchor_pair` (a pair of bodyparts
    that forms the anchor axis -- e.g. body/tail_base on the animal's
    torso). Triangulating from a fixed diagonal avoids the common problem of
    underestimated area when the polygon self-intersects (e.g. during a fast
    head rotation, with ears swapping relative sides) -- which would happen
    using the Shoelace formula directly on an N-vertex polygon with no
    convexity guarantee.
    """
    a1, a2 = anchor_pair
    total_area = 0.0
    for v in vertices:
        if v in anchor_pair:
            continue
        total_area = total_area + _triangle_area(
            df[f'{a1}_x'], df[f'{a1}_y'],
            df[f'{a2}_x'], df[f'{a2}_y'],
            df[f'{v}_x'], df[f'{v}_y'],
        )
    return total_area


def add_rolling_features(df, columns, windows=(5, 15, 30), center=True):
    """
    For each column in `columns`, add `{col}_mean_{w}` and `{col}_std_{w}`
    for every window in `windows`. Returns a new DataFrame (single
    concatenation, avoids the performance fragmentation of inserting one
    column at a time).
    """
    new_cols = {}
    for col in columns:
        for w in windows:
            new_cols[f'{col}_mean_{w}'] = df[col].rolling(window=w, center=center).mean()
            new_cols[f'{col}_std_{w}'] = df[col].rolling(window=w, center=center).std()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def extract_kinematics_features(
    df,
    velocity_bodyparts,
    distance_pairs=(),
    angle_triplets=(),
    polygon_specs=(),
    rolling_windows=(5, 15, 30),
    fillna_value=0,
):
    """
    Generic feature-extraction engine. All parameters are optional lists --
    only pass what makes sense for your bodypart scheme.

    - velocity_bodyparts: list of bodyparts to compute raw velocity for
      (e.g. ['nose', 'head', 'body', 'tail_base']).
    - distance_pairs: list of (bp_a, bp_b) tuples for point-to-point distance.
    - angle_triplets: list of (bp_a, bp_vertex, bp_c) tuples for angles.
    - polygon_specs: list of dicts {'name': str, 'vertices': [...],
      'anchor_pair': (bp_a, bp_b)} for polygon areas.
    - rolling_windows: window sizes (in frames) for smoothing (_mean_W/_std_W)
      of ALL columns generated above.

    Returns a new DataFrame with the original columns plus all features.
    """
    df_features = df.copy()
    columns_to_smooth = []

    for bp in velocity_bodyparts:
        col = f'vel_{bp}'
        df_features[col] = calc_velocity(df_features, bp)
        columns_to_smooth.append(col)

    for bp_a, bp_b in distance_pairs:
        col = f'dist_{bp_a}_{bp_b}'
        df_features[col] = calc_distance(df_features, bp_a, bp_b)
        columns_to_smooth.append(col)

    for bp_a, bp_vertex, bp_c in angle_triplets:
        col = f'{bp_a}_{bp_vertex}_{bp_c}_deg'
        df_features[col] = calc_angle(df_features, bp_a, bp_vertex, bp_c)
        columns_to_smooth.append(col)

    for spec in polygon_specs:
        col = spec['name']
        df_features[col] = calc_polygon_area(df_features, spec['vertices'], spec['anchor_pair'])
        columns_to_smooth.append(col)

    if rolling_windows:
        df_features = add_rolling_features(df_features, columns_to_smooth, windows=rolling_windows)

    if fillna_value is not None:
        df_features = df_features.fillna(fillna_value)

    return df_features
