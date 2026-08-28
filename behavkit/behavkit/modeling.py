"""Multiclass classification (Random Forest) with video-grouped validation."""
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from .utils import logger

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False


def build_feature_list(base_columns, windows=(5, 15, 30), include_std=True, include_raw=True):
    """
    Build the list of feature column names from the base columns (e.g.
    ['vel_nose', 'vel_head']) and the smoothing windows used in
    `extract_kinematics_features` -- avoids repeating this list
    comprehension in every notebook.
    """
    features = []
    for col in base_columns:
        if include_raw:
            features.append(col)
        for w in windows:
            features.append(f'{col}_mean_{w}')
            if include_std:
                features.append(f'{col}_std_{w}')
    return features


def get_or_create_group_holdout(X, y, groups, split_path, n_splits=5, random_state=42):
    """
    Single holdout, stratified by video (`StratifiedGroupKFold`), frozen to
    disk (`split_path`, a .csv with the holdout's video_id values). Reuses
    the same split across runs -- essential for fair comparisons between
    configurations (features, class balancing, etc.), since the split could
    otherwise silently change if `y` changes (e.g. merging labels) and the
    CV is left to re-shuffle.
    """
    if os.path.exists(split_path):
        holdout_videos = set(pd.read_csv(split_path)['video_id'])
        logger.info(f"Split loaded from {split_path} ({len(holdout_videos)} videos in holdout).")
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        _, test_idx = next(sgkf.split(X, y, groups=groups))
        holdout_videos = set(np.asarray(groups)[test_idx])
        pd.Series(sorted(holdout_videos), name='video_id').to_csv(split_path, index=False)
        logger.info(f"Split generated and saved to {split_path} ({len(holdout_videos)} videos in holdout).")

    test_mask = pd.Series(groups).isin(holdout_videos).to_numpy()
    return ~test_mask, test_mask  # train_mask, test_mask


def grid_search_group_cv(X_train, y_train, groups_train, param_grid,
                          n_estimators_search=100, n_splits=5, use_balanced_rf=False,
                          class_weight='balanced_subsample', scoring='f1_macro',
                          random_state=42, n_jobs=-1, verbose=1):
    """
    GridSearchCV with StratifiedGroupKFold -- ensures the inner tuning folds
    respect the groups (whole videos on one side only) and, where possible,
    the class proportions (important for minority classes).

    `use_balanced_rf=True` uses `BalancedRandomForestClassifier`
    (imbalanced-learn: per-tree undersampling) instead of
    `class_weight='balanced_subsample'` -- more effective when reweighting
    impurity alone isn't enough (see the package README for why the two
    techniques aren't equivalent).
    """
    inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if use_balanced_rf:
        if not _HAS_IMBLEARN:
            raise ImportError("BalancedRandomForestClassifier requires 'pip install imbalanced-learn' "
                               "(or install behavkit with the [balanced] extra).")
        rf_base = BalancedRandomForestClassifier(
            random_state=random_state, sampling_strategy='all', replacement=True,
            n_estimators=n_estimators_search, max_features='sqrt', n_jobs=1,
        )
    else:
        rf_base = RandomForestClassifier(
            random_state=random_state, class_weight=class_weight,
            n_estimators=n_estimators_search, max_features='sqrt', n_jobs=1,
        )

    search = GridSearchCV(
        estimator=rf_base, param_grid=param_grid, cv=inner_cv,
        scoring=scoring, n_jobs=n_jobs, verbose=verbose,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search


def train_final_model(best_params, X, y, n_estimators=500, use_balanced_rf=False,
                       class_weight='balanced_subsample', random_state=42, n_jobs=-1):
    """Retrain with the best hyperparameters, using all the data passed in and more trees."""
    if use_balanced_rf:
        if not _HAS_IMBLEARN:
            raise ImportError("BalancedRandomForestClassifier requires 'pip install imbalanced-learn'.")
        model = BalancedRandomForestClassifier(
            **best_params, n_estimators=n_estimators, random_state=random_state,
            sampling_strategy='all', replacement=True, max_features='sqrt', n_jobs=n_jobs,
        )
    else:
        model = RandomForestClassifier(
            **best_params, n_estimators=n_estimators, random_state=random_state,
            class_weight=class_weight, max_features='sqrt', n_jobs=n_jobs,
        )
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Returns (classification_report text, classification_report as a tidy
    DataFrame -- ready for `.to_csv()`, confusion matrix, labels).
    """
    y_pred = model.predict(X_test)
    report_text = classification_report(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return report_text, report_df, cm, labels
