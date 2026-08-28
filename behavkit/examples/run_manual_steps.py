"""
Example using the individual functions directly (no config file), for cases
where you want more control than run_pipeline() gives you. Every step below
that produces a result you'd want to inspect writes it to a plain CSV --
this package does not do any plotting itself.
"""
import numpy as np
import pandas as pd
import behavkit as bk

bk.setup_logging()

# %% 1. Discover (scoring spreadsheet, DLC output) pairs
pairs = bk.discover_video_pairs(
    behavsoft_dir="behavsoft_ethograms",
    dlc_dir="dlc_outputs",
)

# %% 2. Batch preprocessing
# Adjust velocity_bodyparts/distance_pairs/angle_triplets to your DLC model's scheme.
feature_kwargs = dict(
    velocity_bodyparts=['nose', 'head', 'body', 'tail_base'],
    distance_pairs=[('nose', 'tail_base'), ('head', 'body')],
    angle_triplets=[('nose', 'body', 'tail_base')],
    rolling_windows=(5, 15, 30),
)

df_train, df_full = bk.preprocess_batch(
    pairs,
    feature_kwargs=feature_kwargs,
    fps=30,
    label_map=None,  # e.g. {'Rear_raw': 'Rearing', 'Groom_raw': 'Grooming'}
)

# %% 3. Multiclass classification (Random Forest) with video-grouped CV
base_columns = ['vel_nose', 'vel_head', 'vel_body', 'vel_tail_base']
feature_columns = bk.build_feature_list(base_columns, windows=(5, 15, 30))

X = df_train[feature_columns].astype(np.float32)
y = df_train['y_target_behavior']
groups = df_train['video_id']

train_mask, test_mask = bk.get_or_create_group_holdout(
    X, y, groups, split_path="holdout_video_ids.csv"
)
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
groups_train = groups[train_mask]

param_grid = {
    'max_depth': [10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
}

search = bk.grid_search_group_cv(
    X_train, y_train, groups_train, param_grid,
    n_estimators_search=100, use_balanced_rf=False,
)
print(f"Best hyperparameters: {search.best_params_}")

final_model = bk.train_final_model(search.best_params_, X_train, y_train, n_estimators=500)
report_text, report_df, cm, labels = bk.evaluate_model(final_model, X_test, y_test)
print(report_text)

report_df.to_csv("model_classification_report.csv")
pd.DataFrame(cm, index=labels, columns=labels).to_csv("model_confusion_matrix.csv")

# %% 4. Behavior sequence export (for Markov/sequence analysis elsewhere)
df_events = bk.export_behavior_sequences(df_full, "behavior_sequences.csv")
print(df_events.head())

# %% 5. Annotated video using the model's predictions
example_video_id = df_train['video_id'].unique()[0]
df_example = df_full[df_full['video_id'] == example_video_id].sort_values('time_s')
predictions = final_model.predict(df_example[feature_columns].astype(np.float32))

CLASS_COLORS = {
    'Rearing': (0, 200, 0),
    'Grooming': (200, 150, 0),
    'Walking': (200, 0, 0),
    'Immobility': (0, 0, 200),
}

bk.annotate_video_classification(
    video_path=f"raw_videos/{example_video_id}.mp4",
    predictions_per_frame=predictions,
    output_path=f"outputs/videos/{example_video_id}_annotated.mp4",
    class_colors=CLASS_COLORS,
)
