import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import to_html


def get_args():
    parser = argparse.ArgumentParser(description="Confusion Matrix from Output")
    parser.add_argument(
        "--phanns1",
        action="store_true",
        help="Optional flag to just produce the original phanns conf. matrix with these settings.",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="Path to the input CSV file (PhANNs output file).",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=False,
        help="Path to the output HTML file for the confusion matrix.",
    )

    return parser.parse_args()


def plot(array, classes):
    color_scale = [
        (0.0, "rgb(13, 8, 135)"),  # Deep blue
        # (0.1, "rgb(82, 1, 163)"),    # Purple-blue
        (0.1, "rgb(150, 0, 167)"),  # Purple
        (0.2, "rgb(188, 20, 155)"),  # Purple-red
        (0.3, "rgb(213, 52, 139)"),  # Magenta
        (0.4, "rgb(240, 78, 110)"),  # Pinkish-red
        (0.5, "rgb(254, 103, 80)"),  # Orange-red
        (0.6, "rgb(254, 145, 60)"),  # Orange
        (0.7, "rgb(254, 188, 50)"),  # Yellow-orange
        (0.8, "rgb(253, 220, 46)"),  # Yellow
        (0.9, "rgb(252, 236, 44)"),  # Light yellow
        (1.0, "rgb(252, 252, 43)"),  # Bright yellow
    ]

    fig = px.imshow(
        array,
        x=list(classes.keys()),
        y=list(classes.keys()),
        title="Confusion Matrix - Recall",
        text_auto=".2f",
        color_continuous_scale=color_scale,
    ).update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        title={"x": 0.1},
        width=800,
        height=800,
        title_font=dict(size=28),  # Increase the font size of the title
        xaxis=dict(
            title_font=dict(size=20)
        ),  # Increase the font size for the x-axis label
        yaxis=dict(
            title_font=dict(size=20)
        ),  # Increase the font size for the y-axis label
    )
    fig.update_layout(font_family="sans-serif")

    return fig


def phanns1():
    output_file = "2019_phanns1.html"
    array = np.array(
        [
            [
                0.74,
                0.01,
                0.10,
                0.0,
                0.0,
                0.10,
                0.02,
                0.03,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.84,
                0.01,
                0.0,
                0.0,
                0.03,
                0.02,
                0.09,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.01,
                0.0,
                0.84,
                0.0,
                0.0,
                0.08,
                0.06,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.92,
                0.0,
                0.04,
                0.02,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.10,
                0.77,
                0.02,
                0.0,
                0.08,
                0.0,
                0.01,
                0.01,
            ],
            [
                0.01,
                0.0,
                0.02,
                0.02,
                0.01,
                0.87,
                0.02,
                0.03,
                0.0,
                0.01,
                0.01,
            ],
            [
                0.0,
                0.0,
                0.01,
                0.01,
                0.0,
                0.14,
                0.81,
                0.0,
                0.0,
                0.03,
                0.0,
            ],
            [
                0.02,
                0.05,
                0.0,
                0.0,
                0.01,
                0.13,
                0.0,
                0.76,
                0.0,
                0.0,
                0.03,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.01,
                0.01,
                0.02,
                0.01,
                0.01,
                0.94,
                0.0,
                0.0,
            ],
            [
                0.01,
                0.01,
                0.0,
                0.0,
                0.0,
                0.07,
                0.12,
                0.0,
                0.01,
                0.77,
                0.0,
            ],
            [
                0.01,
                0.0,
                0.01,
                0.01,
                0.01,
                0.07,
                0.01,
                0.05,
                0.01,
                0.01,
                0.82,
            ],
        ]
    )

    classes = {
        "BPL": 0,
        "CLR": 1,
        "HTJ": 2,
        "MCP": 3,
        "MTL": 4,
        "OTH": 5,
        "PTL": 6,
        "TFR": 7,
        "TSH": 9,
        "m_CP": 10,
        "m_TL": 11,
    }

    fig = plot(array, classes)

    write(output_file, fig)


def confusion_matrix(file_path, true_class, predicted_class):
    all_classes = list(set(true_class).union(set(predicted_class)))
    classes = {x: i for i, x in enumerate(sorted(all_classes))}

    matrix = np.zeros((len(classes), len(classes)))
    for true, pred in zip(true_class, predicted_class):
        matrix[classes[true]][classes[pred]] += 1

    row_sums = matrix.sum(axis=1)
    normalized_matrix = matrix / row_sums[:, np.newaxis]

    fig = plot(normalized_matrix, classes)

    write(file_path, fig)


def write(output_file, fig):
    with open(output_file, "w") as output:
        output.write(to_html(fig, include_plotlyjs="cdn"))
    print(f"Confusion matrix graph written to {Path(output_file).absolute()}")


if __name__ == "__main__":
    args = get_args()

    input_file = args.input_file
    output_file = args.output_file
    if not output_file:
        output_file = input_file.replace(".csv", ".html")

    if args.phanns1:
        phanns1(output_file)
        exit(0)

    # Read the CSV file
    df = pd.read_csv(input_file)

    if "true_class" not in df.columns or "prediction" not in df.columns:
        raise ValueError(
            "The input file must contain 'true_class' and 'prediction' columns."
        )

    # Extract true and predicted classes
    true_class = df["true_class"].tolist()
    predicted_class = df["prediction"].tolist()

    # Generate confusion matrix
    confusion_matrix(output_file, true_class, predicted_class)
