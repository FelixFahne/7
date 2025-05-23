#!/usr/bin/env python
"""Utilities for additional ESL experiments.

This module merges code from the original notebook and script and provides
helper functions for loading the data, feature extraction as well as several
training utilities used in the experiments. The heavy in-notebook execution
cells were removed so that this file can be imported as a library or executed
from the command line if desired.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import tensorflow as tf


# ---------------------------------------------------------------------------
# helper functions for label handling and scoring
# ---------------------------------------------------------------------------


def transform(x: str) -> str:
    """Replace ``Unknown`` label level with ``Dialogue level``."""
    return "Dialogue level" if x == "Unknown" else x


def assign_tone(row: pd.Series) -> str:
    """Assign a dialogue tone based on several linguistic features."""
    if (
        row["backchannels"] > 0
        or row["code-switching for communicative purposes"] > 0
        or row["collaborative finishes"] > 0
    ):
        return "Informal"
    if (
        row["subordinate clauses"] > 0
        or row["impersonal subject + non-factive verb + NP"] > 0
    ):
        return "Formal"
    return "Neutral"


def get_new_labels(
    counts_df: pd.DataFrame,
    new_column: str = "P-V",
    old_column: Iterable[str] = (
        "topic extension with the same content",
        "topic extension under the previous direction",
    ),
) -> pd.DataFrame:
    """Concatenate two columns into a combined label column."""
    cc_dd = counts_df[list(old_column)].astype(int).values
    cc_dd = ["-".join(str(j) for j in row) for row in cc_dd]
    counts_df[new_column] = cc_dd
    return counts_df


def get_result(test_y: Iterable, test_pred: Iterable) -> List[float]:
    """Return classification metrics (accuracy, precision, recall, F1)."""
    acc = metrics.accuracy_score(test_y, test_pred)
    pre = metrics.precision_score(test_y, test_pred, average="weighted")
    rec = metrics.recall_score(test_y, test_pred, average="weighted")
    f1s = metrics.f1_score(test_y, test_pred, average="weighted")
    return [acc, pre, rec, f1s]


def get_result_2(test_y: Iterable, test_pred: Iterable) -> List[float]:
    """Return regression metrics (MAE, MSE, MAPE, R2)."""
    mae = metrics.mean_absolute_error(test_y, test_pred)
    mse = metrics.mean_squared_error(test_y, test_pred)
    mape = metrics.mean_absolute_percentage_error(test_y, test_pred)
    r2 = metrics.r2_score(test_y, test_pred)
    return [mae, mse, mape, r2]


def get_normal(
    data: pd.DataFrame,
    token_counts: Dict[str, int],
    utterance_counts: Dict[str, int],
) -> pd.DataFrame:
    """Compute normalised counts used in the original experiments."""
    normal_list: List[float] = []
    for _, label, level, content in data.values:
        if level == "Token level":
            norm = len(content.split("&&&&")[1].split()) / token_counts[label]
        elif level == "Utterance level":
            norm = len(content.split("&&&&")[1].split()) / utterance_counts[label]
        else:
            norm = -1
        normal_list.append(norm)
    data["normal"] = normal_list
    return data


score_dict = {
    "topic extension with clear new context": [5, "Y1"],
    "topic extension under the previous direction": [4, "Y1"],
    "topic extension with the same content": [3, "Y1"],
    "repeat and no topic extension": [2, "Y1"],
    "no topic extension and stop the topic at this point": [1, "Y1"],
    "overall tone choice: very informal": [5, "Y2"],
    "overall tone choice: quite informal, but some expressions are still formal": [
        4,
        "Y2",
    ],
    "overall tone choice: quite formal and some expressions are not that formal": [
        3,
        "Y2",
    ],
    "overall tone choice: very formal": [1, "Y2"],
    "Co1": [5, "Y3"],
    "Co2": [4, "Y3"],
    "Co3": [3, "Y3"],
    "Co4": [2, "Y3"],
    "Co5": [1, "Y3"],
    "Cc1": [5, "Y4"],
    "Cc2": [4, "Y4"],
    "Cc3": [3, "Y4"],
    "Cc4": [2, "Y4"],
    "Cc5": [1, "Y4"],
    "conversation opening": [3, "Y3"],
    "conversation closing": [3, "Y4"],
}


# ---------------------------------------------------------------------------
# data loading and feature preparation
# ---------------------------------------------------------------------------


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load all CSV files in ``data_path`` and fix label levels."""
    frames = []
    for fname in os.listdir(data_path):
        fp = os.path.join(data_path, fname)
        if not fp.lower().endswith(".csv"):
            continue
        frames.append(pd.read_csv(fp))
    data = pd.concat(frames, ignore_index=True)

    new_labellevel: List[str] = []
    for _, label, level, _ in data.values:
        if level == "Unknown":
            if label == "adjectives/ adverbs expressing possibility":
                new_v = "Utterance level"
            else:
                new_v = "Dialogue level"
        else:
            new_v = level
        new_labellevel.append(new_v)

    data["LabelLevel"] = new_labellevel
    data.index = range(len(data))
    return data


def compute_label_counts(
    data: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], List[str]]:
    """Return frequency dictionaries and the list of valid feature keys."""
    token_counts = collections.Counter(
        data[data["LabelLevel"] == "Token level"]["Label"]
    )
    utterance_counts = collections.Counter(
        data[data["LabelLevel"] == "Utterance level"]["Label"]
    )
    dialogue_counts = collections.Counter(
        data[data["LabelLevel"] == "Dialogue level"]["Label"]
    )
    valid_keys = list(utterance_counts.keys()) + list(token_counts.keys())
    return token_counts, utterance_counts, dialogue_counts, valid_keys


def extract_features(
    data: pd.DataFrame,
    token_counts: Dict[str, int],
    utterance_counts: Dict[str, int],
    valid_keys: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract feature matrices used in the experiments."""
    all_people_df = pd.DataFrame()
    all_labels_df = pd.DataFrame()
    all_classs_df = pd.DataFrame()

    for index in range(0, len(data), 10):
        index_data = data.iloc[index : index + 10].copy()
        index_data["word_num"] = [
            len(val.split("&&&&")[1].split()) for val in index_data["Content"]
        ]

        test_feature: Dict[str, float] = {}
        for label in index_data["Label"].unique():
            index_data_label = index_data[index_data["Label"] == label]
            cnt = pd.DataFrame(
                [collections.Counter(index_data_label["word_num"])], index=[label]
            ) / len(index_data_label)
            col = cnt.columns.to_numpy(dtype=float)
            test_feature[label] = (col * cnt.values[0]).sum() / cnt.sum().sum()
        test_feature_df = pd.DataFrame([test_feature], index=[index])
        all_people_df = pd.concat([all_people_df, test_feature_df])

        new_score_list = [
            score_dict.get(lb, [None, None]) for lb in index_data["Label"]
        ]
        index_data[["score", "type"]] = new_score_list

        type_score: Dict[str, float] = {}
        type_class: Dict[str, int] = {}
        for type_ in ["Y1", "Y2", "Y3", "Y4"]:
            type_data = index_data[index_data["type"] == type_]
            if len(type_data) == 0:
                type_score[type_] = 3
            else:
                type_score[type_] = type_data["score"].mean()

            type_data_big3 = type_data[type_data["score"] > 3]
            type_data_sma3 = type_data[type_data["score"] < 3]
            if len(type_data_big3) + len(type_data_sma3) == 0:
                type_class[type_] = 3
            else:
                if len(type_data_sma3) > 0:
                    type_data_sma3 = type_data_sma3.sort_values(
                        by=["score", "word_num"]
                    )
                    best_score = type_data_sma3.values[0, -2]
                    type_data_sma3 = type_data_sma3[
                        type_data_sma3["score"] == best_score
                    ]
                    best_word = type_data_sma3.values[-1, -3]
                    type_data_sma3 = type_data_sma3[
                        type_data_sma3["word_num"] == best_word
                    ]
                if len(type_data_big3) > 0:
                    type_data_big3 = type_data_big3.sort_values(
                        by=["score", "word_num"]
                    )
                    best_score = type_data_big3.values[-1, -2]
                    type_data_big3 = type_data_big3[
                        type_data_big3["score"] == best_score
                    ]
                    best_word = type_data_big3.values[-1, -3]
                    type_data_big3 = type_data_big3[
                        type_data_big3["word_num"] == best_word
                    ]
                type_concat = pd.concat([type_data_sma3, type_data_big3])
                type_concat = type_concat.sort_values(by=["word_num"])
                type_class[type_] = int(type_concat.values[-1, -2])

        type_score_df = pd.DataFrame([type_score], index=[index])
        all_labels_df = pd.concat([all_labels_df, type_score_df])
        type_class_df = pd.DataFrame([type_class], index=[index])
        all_classs_df = pd.concat([all_classs_df, type_class_df])

    data_x = all_people_df[valid_keys]
    data_x = data_x.fillna(0)
    data_y = all_labels_df.copy()
    data_class = all_classs_df.copy()
    return data_x, data_y, data_class


# ---------------------------------------------------------------------------
# model utilities
# ---------------------------------------------------------------------------

model_dict = {
    "Linear": LinearRegression(),
    "Logit": LogisticRegression(),
    "RF_R": RandomForestRegressor(),
    "RF_C": RandomForestClassifier(),
    "NB_R": BayesianRidge(),
    "NB_C": BernoulliNB(),
}

topk = 10


def save_csv(data: pd.DataFrame, path: str) -> None:
    """Write ``data`` to ``path`` as UTF-8 CSV."""
    data.to_csv(f"./{path}.csv", encoding="utf_8_sig")


def get_all_result(
    train_x_normal: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x_normal: pd.DataFrame,
    test_y: pd.DataFrame,
    model_name: str,
):
    """Train a classical model and return predictions and metrics."""
    test_pred_dict: Dict[str, np.ndarray] = {}
    test_true_dict: Dict[str, pd.Series] = {}
    train_pred_dict: Dict[str, np.ndarray] = {}
    train_true_dict: Dict[str, pd.Series] = {}
    train_result_linear: Dict[str, List[float]] = {}
    test_result_linear: Dict[str, List[float]] = {}
    model_infortance = pd.DataFrame()

    for co in train_y.columns:
        train_true = train_y[co]
        test_true = test_y[co]
        model = model_dict[model_name]
        model.fit(train_x_normal, train_true)

        train_pred = model.predict(train_x_normal)
        test_pred = model.predict(test_x_normal)

        train_pred_dict[co] = train_pred if len(train_pred) == 1 else train_pred[0]
        test_pred_dict[co] = test_pred if len(test_pred) == 1 else test_pred[0]
        train_true_dict[co] = train_true
        test_true_dict[co] = test_true

        if model_name in ["Linear", "NB_R", "RF_R"]:
            result_func = get_result_2
            metrics_list = ["mae", "mse", "mape", "r2"]
        else:
            result_func = get_result
            metrics_list = ["acc", "pre", "rec", "f1s"]
        train_result_linear[co] = result_func(train_true, train_pred)
        test_result_linear[co] = result_func(test_true, test_pred)

        importance = None
        if model_name in ["Linear", "NB_R"]:
            importance = pd.DataFrame(
                model.coef_, index=model.feature_names_in_, columns=[co]
            )
        elif model_name in ["RF_C", "RF_R"]:
            importance = pd.DataFrame(
                model.feature_importances_, index=model.feature_names_in_, columns=[co]
            )
        elif model_name in ["Logit", "NB_C"]:
            importance = pd.DataFrame(
                model.coef_[0], index=model.feature_names_in_, columns=[co]
            )

        if importance is not None:
            importance_ = importance.abs().sort_values(by=[co])
            importance = importance.loc[importance_.index[-topk:]]
            model_infortance = pd.concat([model_infortance, importance], axis=1)

    train_result = pd.DataFrame(train_result_linear, index=metrics_list)
    test_result = pd.DataFrame(test_result_linear, index=metrics_list)
    test_pred_df = pd.DataFrame(test_pred_dict, index=test_y.index)
    test_true_df = pd.DataFrame(test_true_dict, index=test_y.index)
    train_pred_df = pd.DataFrame(train_pred_dict, index=train_y.index)
    train_true_df = pd.DataFrame(train_true_dict, index=train_y.index)
    return (
        [train_result, test_result],
        [test_true_df, test_pred_df],
        [train_true_df, train_pred_df],
        model_infortance,
    )


# ---------------------------------------------------------------------------
# simple neural network models
# ---------------------------------------------------------------------------


def get_model_R(input_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    dense = tf.keras.layers.Dense(64, activation="relu")(inputs)
    dense = tf.keras.layers.Dense(64, activation="relu")(dense)
    outputs = tf.keras.layers.Dense(output_dim, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def get_model_C(input_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    dense = tf.keras.layers.Dense(64, activation="relu")(inputs)
    dense = tf.keras.layers.Dense(64, activation="relu")(dense)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(dense)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def get_BP_result(
    train_x_normal: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x_normal: pd.DataFrame,
    test_y: pd.DataFrame,
    model_name: str,
):
    input_dim = train_x_normal.shape[1]
    output_dim = train_y.shape[1]

    if model_name == "BP_R":
        train_y_normal = train_y / 5
        model_bp = get_model_R(input_dim, output_dim)
    else:
        train_y_normal = train_y.copy()
        model_bp = get_model_C(input_dim, output_dim)

    stopearly = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, monitor="val_loss"
    )
    model_bp.fit(
        train_x_normal,
        train_y_normal,
        validation_split=0.1,
        batch_size=16,
        epochs=100,
        callbacks=[stopearly],
        verbose=0,
    )

    test_pred_bp = model_bp.predict(test_x_normal)
    train_pred_bp = model_bp.predict(train_x_normal)

    if model_name == "BP_R":
        test_pred_bp *= 5
        train_pred_bp *= 5
    else:
        train_pred_bp = (train_pred_bp >= 0.5).astype(int)
        test_pred_bp = (test_pred_bp >= 0.5).astype(int)

    test_pred_bp = pd.DataFrame(
        test_pred_bp, index=test_y.index, columns=test_y.columns
    )
    train_pred_bp = pd.DataFrame(
        train_pred_bp, index=train_y.index, columns=train_y.columns
    )

    train_result_dict: Dict[str, List[float]] = {}
    test_result_dict: Dict[str, List[float]] = {}

    if model_name in ["BP_R"]:
        result_func = get_result_2
        metrics_list = ["mae", "mse", "mape", "r2"]
    else:
        result_func = get_result
        metrics_list = ["acc", "pre", "rec", "f1s"]

    for co in train_pred_bp.columns:
        true_train = train_y[co]
        pred_train = train_pred_bp[co]
        true_test = test_y[co]
        pred_test = test_pred_bp[co]

        train_result_dict[co] = result_func(true_train, pred_train)
        test_result_dict[co] = result_func(true_test, pred_test)

    train_result = pd.DataFrame(train_result_dict, index=metrics_list)
    test_result = pd.DataFrame(test_result_dict, index=metrics_list)
    return (
        [train_result, test_result],
        [test_y, test_pred_bp],
        [train_y, train_pred_bp],
        None,
    )


def get_BP_result_C(
    train_x_normal: pd.DataFrame,
    train_c: pd.DataFrame,
    test_x_normal: pd.DataFrame,
    test_c: pd.DataFrame,
    model_name: str,
):
    input_dim = train_x_normal.shape[1]
    output_dim = train_c.shape[1]

    result_func = get_result
    metrics_list = ["acc", "pre", "rec", "f1s"]

    train_pred_dict: Dict[str, np.ndarray] = {}
    test_pred_dict: Dict[str, np.ndarray] = {}
    train_result_dict: Dict[str, List[float]] = {}
    test_result_dict: Dict[str, List[float]] = {}

    for co in train_c.columns:
        train_y_normal = train_c[[co]] - 1
        test_y_normal = test_c[[co]] - 1
        model_bp = get_model_C(input_dim, output_dim)
        stopearly = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        )
        model_bp.fit(
            train_x_normal,
            train_y_normal,
            validation_split=0.1,
            batch_size=16,
            epochs=100,
            callbacks=[stopearly],
            verbose=0,
        )

        test_pred_bp = model_bp.predict(test_x_normal)
        train_pred_bp = model_bp.predict(train_x_normal)

        train_pred_bp = train_pred_bp.argmax(1) + 1
        test_pred_bp = test_pred_bp.argmax(1) + 1
        train_pred_dict[co] = train_pred_bp
        test_pred_dict[co] = test_pred_bp

        train_result_dict[co] = result_func(train_y_normal + 1, train_pred_bp)
        test_result_dict[co] = result_func(test_y_normal + 1, test_pred_bp)

    train_result = pd.DataFrame(train_result_dict, index=metrics_list)
    test_result = pd.DataFrame(test_result_dict, index=metrics_list)

    train_pred_bp = pd.DataFrame(train_pred_dict, index=train_c.index)
    test_pred_bp = pd.DataFrame(test_pred_dict, index=test_c.index)
    return (
        [train_result, test_result],
        [test_c, test_pred_bp],
        [train_c, train_pred_bp],
        None,
    )


# ---------------------------------------------------------------------------
# statistical evaluation helpers
# ---------------------------------------------------------------------------


def get_t_test_result(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Perform a t-test between true and predicted values for each model."""
    from scipy import stats

    t_test_result: Dict[str, Dict[str, float]] = {}
    for model_n, (true, pred) in results.items():
        type_t: Dict[str, float] = {}
        for col in true.columns:
            t, pval = stats.ttest_ind(true[col].values, pred[col].values)
            type_t[col] = pval
        t_test_result[model_n] = type_t
    return pd.DataFrame(t_test_result)


# ---------------------------------------------------------------------------
# convenience routine
# ---------------------------------------------------------------------------


def train_test_split_normalized(
    data_x: pd.DataFrame,
    data_y: pd.DataFrame,
    data_class: pd.DataFrame,
    train_size: float = 0.8,
):
    """Split the data and return z-normalised feature matrices."""
    train_x, test_x, train_y_split, test_y_split, train_c, test_c = train_test_split(
        data_x, data_y, data_class, train_size=train_size
    )
    mean = train_x.mean()
    std = train_x.std()
    train_x_normal = (train_x - mean) / std
    test_x_normal = (test_x - mean) / std
    return train_x_normal, test_x_normal, train_y_split, test_y_split, train_c, test_c


if __name__ == "__main__":
    data_dir = Path("./data_csv_sample")
    if data_dir.exists():
        data = load_dataset(data_dir)
        token_counts, utterance_counts, dialogue_counts, valid_keys = (
            compute_label_counts(data)
        )
        data_x, data_y, data_class = extract_features(
            data, token_counts, utterance_counts, valid_keys
        )
        (
            train_x_normal,
            test_x_normal,
            train_y_split,
            test_y_split,
            train_c,
            test_c,
        ) = train_test_split_normalized(data_x, data_y, data_class)
        results = get_all_result(
            train_x_normal, train_y_split, test_x_normal, test_y_split, "Linear"
        )
        print(results[0][1])
    else:
        print(f"Data directory {data_dir!s} not found. No action taken.")
