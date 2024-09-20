from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm
from utils.stored_models import get_model_dir


def write_parquet_table(
    model_name: str,
    tp_gaussian_weights_dict: dict,
    fp_gaussian_weights_dict: dict,
    confidence_scores: dict,
):
    saved_model_dir = get_model_dir(model_name)

    confidence_scores_dir = saved_model_dir / "confidence_scores"
    confidence_scores_dir.mkdir(exist_ok=True, parents=True)

    tp_table = pa.table(tp_gaussian_weights_dict)
    fp_table = pa.table(fp_gaussian_weights_dict)
    confidence_scores_table = pa.table(confidence_scores)

    pq.write_table(tp_table, confidence_scores_dir / "tp_weights.parquet")
    pq.write_table(fp_table, confidence_scores_dir / "fp_weights.parquet")
    pq.write_table(
        confidence_scores_table,
        confidence_scores_dir / "class_confidence_scores.parquet",
    )


def get_TP_scores(df: pd.DataFrame, class_name: str):
    return df[(df["prediction"] == class_name) & (df["true_class"] == class_name)][
        class_name
    ].values


def get_FP_scores(df: pd.DataFrame, class_name: str):
    return df[(df["prediction"] == class_name) & (df["true_class"] != class_name)][
        class_name
    ].values


def calculate_point_weights(scores, radius):
    x_min = 0
    x_max = 10
    step = 0.01

    # Create an array for all x values from x_min to x_max with increments of step
    x_values = np.arange(x_min + step, x_max + step, step)

    # Reshape scores and x_values to allow for broadcasting
    scores = np.array(scores)
    x_values = x_values[:, np.newaxis]

    # Compute the Gaussian contributions for all x and scores at once
    weights = norm.pdf(x_values, loc=scores, scale=radius)

    # Sum the weights along the score axis (columns)
    return np.sum(weights, axis=1), x_values.flatten()


def count_sign_changes(weights):
    previous_weight = next(iter(weights))
    sign_changes = 0
    cursor = 0
    sign_positive = True
    for next_weight in islice(weights, 1, len(weights) - 1):
        if next_weight - previous_weight > 0:
            if not sign_positive:
                sign_changes += 1
            sign_positive = True
        elif next_weight - previous_weight < 0:
            if sign_positive:
                sign_changes += 1
            sign_positive = False
        previous_weight = next_weight
        cursor += 1

    return sign_changes


def calculate_confidence_scores(tp_gaussian_weights, fp_gaussian_weights, x_values):
    confidence_scores = []

    for x in range(len(x_values)):
        tp_score = tp_gaussian_weights[x]
        fp_score = fp_gaussian_weights[x]
        confidence_scores.append(tp_score / (tp_score + fp_score))

    return confidence_scores


def gaussian_confidences(predictions_path: Path, model_name: str):
    assert predictions_path.exists(), f"File not found: {predictions_path}"

    df = pd.read_csv(predictions_path)
    classes = list(df.columns[1:-1])
    tp_gaussian_weights_dict = {}
    fp_gaussian_weights_dict = {}
    confidence_scores = {}

    for class_name in classes:
        radius = 0.05
        while True:
            tp_scores = get_TP_scores(df, class_name)

            tp_gaussian_weights, x_values = calculate_point_weights(tp_scores, radius)

            sign_changes = count_sign_changes(tp_gaussian_weights)
            title_str = f"TP -- class: {class_name:<6}radius: {radius:<6.3f} sign_changes: {sign_changes}"
            radius += 0.005
            if sign_changes <= 2:
                # px.line(
                #     x=x_values,
                #     y=tp_gaussian_weights,
                #     width=600,
                #     height=600,
                #     title=title_str,
                # ).show()
                tp_gaussian_weights_dict[class_name] = tp_gaussian_weights
                print(title_str)
                break

        while True:
            fp_scores = get_FP_scores(df, class_name)

            fp_gaussian_weights, x_values = calculate_point_weights(fp_scores, radius)

            sign_changes = count_sign_changes(fp_gaussian_weights)
            title_str = f"FP -- class: {class_name:<6}radius: {radius:<6.3f} sign_changes: {sign_changes}"
            radius += 0.005
            if sign_changes <= 2:
                # px.line(
                #     x=x_values,
                #     y=fp_gaussian_weights,
                #     width=600,
                #     height=600,
                #     title=title_str,
                # ).show()
                fp_gaussian_weights_dict[class_name] = fp_gaussian_weights
                print(title_str)
                break

        confidence_scores[class_name] = calculate_confidence_scores(
            tp_gaussian_weights, fp_gaussian_weights, x_values
        )
        # px.scatter(
        #     x=list(range(len(confidence_scores[class_name]))),
        #     y=confidence_scores[class_name],
        #     width=600,
        #     height=600,
        #     title = f"{class_name} confidence scores"
        #     ).show()

    write_parquet_table(
        model_name,
        tp_gaussian_weights_dict,
        fp_gaussian_weights_dict,
        confidence_scores,
        Path("testing/"),
    )
