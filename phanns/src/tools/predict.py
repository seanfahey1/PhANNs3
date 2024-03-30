from pathlib import Path
from utils.data_handler import Data, fasta_count

import numpy as np
import re
from Bio import SeqIO
from tqdm import tqdm
import sys
sys.path.append("..")

from utils import stored_models
from utils.data_handler import Data, fasta_count
from tensorflow.keras.models import load_model


def load_dataset(fasta_path, mean_arr, std_arr):

    # init Data object
    num_proteins = fasta_count([Path(fasta_path)])
    data = Data(num_proteins)
    row_counter = 0

    records = SeqIO.parse(Path(fasta_path), "fasta")
    num_proteins_current_file = fasta_count([Path(fasta_path)])
    for _ in tqdm(range(num_proteins_current_file)):
        record = next(records)
        sequence = record.seq.__str__().upper()
        row = data.feature_extract(sequence)
        data.add_to_array(row, row_counter, -1, -1)

        row_counter += 1

    zscore_array = z_score_from_pre_calculated(data.arr, std_arr, mean_arr)

    return zscore_array


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


def predict(model_name, test_X):
    y_hats = []

    stored_model_dir = stored_models.get_model_dir(model_name) / "model_files/"

    for model_number in range(1, 11):
        model_full_name = f"{'{:02d}'.format(model_number)}.keras"
        model_path = stored_model_dir / model_full_name
        model = load_model(model_path)

        y_hat = model.predict(test_X, verbose=2)
        y_hats.append(y_hat)

    y_hats_array = np.array(y_hats)
    predicted_Y = np.sum(y_hats, axis=0)
    predicted_Y_index = np.argmax(predicted_Y, axis=1)

    return predicted_Y, predicted_Y_index
