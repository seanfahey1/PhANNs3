import logging
import re
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import torch
from Bio import SeqIO
from plotly.io import to_html
from tqdm import tqdm
from utils.data_handler import Data, fasta_count

sys.path.append("..")

# from tensorflow.keras.backend import clear_session
# from tensorflow.keras.models import load_model
from tools.model import SequentialNN
from utils import stored_models
from utils.data_handler import Data, fasta_count


def load_dataset(fasta_path, mean_arr, std_arr):
    print("Loading fasta file")

    # init Data object
    num_proteins = fasta_count([Path(fasta_path)])
    data = Data(num_proteins)
    row_counter = 0
    fasta_headers = []

    records = SeqIO.parse(Path(fasta_path), "fasta")
    num_proteins_current_file = fasta_count([Path(fasta_path)])
    for _ in tqdm(range(num_proteins_current_file)):
        record = next(records)
        sequence = record.seq.__str__().upper()

        row = data.feature_extract(sequence)
        data.add_to_array(row, row_counter, -1, -1)

        fasta_headers.append(record.description)

        row_counter += 1

    zscore_array = z_score_from_pre_calculated(data.arr, std_arr, mean_arr)

    return zscore_array, fasta_headers


def z_score_from_pre_calculated(data, stdev_arr, mean_arr):
    for col_num in range(data.shape[1]):
        stdev_val = stdev_arr[col_num]
        mean_val = mean_arr[col_num]

        for row_num in range(data.shape[0]):
            if stdev_val != 0:
                val = data[row_num, col_num]
                z_val = (val - mean_val) / stdev_val
            else:
                z_val = 0
            data[row_num, col_num] = z_val

    return data


def predict_pytorch(model_name, test_X, model_sizes):
    # set up cuda
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    y_hats = []
    test_X_torch = torch.FloatTensor(test_X).to(device)

    stored_model_dir = stored_models.get_model_dir(model_name) / "model_files/"
    print("Calculating predictions")

    for model_number in tqdm(range(1, 11)):
        feature_count, num_classes = model_sizes[model_number]

        assert torch.cuda.is_available()
        device = torch.device("cuda")

        model_full_name = f"{'{:02d}'.format(model_number)}.pt"
        model_path = stored_model_dir / model_full_name

        model = SequentialNN(feature_count, num_classes)
        model.load_state_dict(torch.load(model_path))

        model.to(device)
        model.eval()

        with torch.no_grad():
            y_hat = model(test_X_torch)
        y_hats.append(y_hat.cpu())

        del model
        del y_hat
        torch.cuda.empty_cache()

    predicted_Y = np.sum(y_hats, axis=0)
    predicted_Y_index = np.argmax(predicted_Y, axis=1)

    return predicted_Y, predicted_Y_index


# def predict(model_name, test_X):
#     logging.getLogger("tensorflow").setLevel(logging.ERROR)
#     y_hats = []

#     stored_model_dir = stored_models.get_model_dir(model_name) / "model_files/"
#     print("Calculating predictions")

#     for model_number in tqdm(range(1, 11)):
#         model_full_name = f"{'{:02d}'.format(model_number)}.keras"
#         model_path = stored_model_dir / model_full_name
#         model = load_model(model_path)

#         y_hat = model.predict(test_X, verbose=0)
#         y_hats.append(y_hat)

#         clear_session()

#     predicted_Y = np.sum(y_hats, axis=0)
#     predicted_Y_index = np.argmax(predicted_Y, axis=1)

#     return predicted_Y, predicted_Y_index


def write_prediction_outputs(
    output_file, predicted_Y, predicted_class, fasta_headers, sorted_group_names
):
    out_data = zip(fasta_headers, predicted_Y, predicted_class)
    with open(output_file, "w") as out:
        out.write(f"header,{','.join(sorted_group_names)},prediction\n")
        for line in out_data:
            description_clean = re.sub(",", " ", line[0])
            out.write(
                f"{description_clean},{','.join(['{:.4f}'.format(x) for x in line[1]])},{line[2]}\n"
            )
    print(f"Predictions written to {Path(output_file).absolute()}")


def write_initial_prediction_outputs(
    output_file, predicted_Y, predicted_class, sorted_group_names, true_class
):
    out_data = zip(true_class, predicted_Y, predicted_class)
    with open(output_file, "w") as out:
        out.write(f"true_class,{','.join(sorted_group_names)},prediction\n")
        for line in out_data:
            out.write(
                f"{line[0]},{','.join(['{:.4f}'.format(x) for x in line[1]])},{line[2]}\n"
            )
    print(f"Initial predictions written to {Path(output_file).absolute()}")


def confusion_matrix(file_path, true_class, predicted_class):
    all_classes = list(set(true_class).union(set(predicted_class)))
    classes = {x: i for i, x in enumerate(sorted(all_classes))}

    matrix = np.zeros((len(classes), len(classes)))
    for true, pred in zip(true_class, predicted_class):
        matrix[classes[true]][classes[pred]] += 1

    row_sums = matrix.sum(axis=1)
    normalized_matrix = matrix / row_sums[:, np.newaxis]

    fig = px.imshow(
        normalized_matrix,
        x=list(classes.keys()),
        y=list(classes.keys()),
        title="Confusion Matrix - Recall",
    ).update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
    )

    with open(file_path, "w") as output:
        output.write(to_html(fig, include_plotlyjs="cdn"))
    print(f"Confusion matrix graph written to {Path(file_path).absolute()}")
