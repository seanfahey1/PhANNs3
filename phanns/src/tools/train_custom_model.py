import logging
import re
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

sys.path.append("..")
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from utils import calc, stored_models
from utils.data_handler import Data, fasta_count


def load_dataset(fasta_dir):
    if not Path(fasta_dir).exists():
        raise ValueError(f"Selected directory ({fasta_dir}) does not exist!")

    # collect files
    fastas = list(Path(fasta_dir).glob("*.fasta"))

    if len(fastas) == 0:
        raise ValueError(f"No .fasta files found at target directory: {fasta_dir}")

    # assign a numeric code to each base file name
    re_pattern = r"(?P<prefix>\d+)_(?P<name>.*)"
    group_names = set()
    for file in fastas:
        group_names.add(re.match(re_pattern, file.stem).group("name"))
    sorted_group_names = sorted(list(group_names))

    class_number_assignments = {x: i for i, x in enumerate(sorted_group_names)}

    # init Data object
    num_proteins = fasta_count(fastas)
    data = Data(num_proteins)
    row_counter = 0

    # establish empty group and class arrays
    group_arr = np.zeros(num_proteins, dtype=int)
    class_arr = np.zeros(num_proteins, dtype=int)

    for file_path in fastas:
        match = re.search(re_pattern, file_path.stem)
        cls = match.group("name")
        group_number = match.group("prefix")
        cls_number = class_number_assignments[cls]

        print(file_path)
        records = SeqIO.parse(file_path, "fasta")
        num_proteins_current_file = fasta_count([file_path])
        for _ in tqdm(range(num_proteins_current_file)):
            record = next(records)
            sequence = record.seq.__str__().upper()
            row = data.feature_extract(sequence)
            data.add_to_array(row, row_counter, cls_number, group_number)

            group_arr[row_counter] = group_number
            class_arr[row_counter] = cls_number

            row_counter += 1

    print("Calculating z-score normalization")
    mean_array, stdev_array, zscore_array = calc.zscore(data.arr)

    return (
        mean_array,
        stdev_array,
        zscore_array,
        group_arr,
        class_arr,
        sorted_group_names,
    )


def train_new_model(name, class_arr, group_arr, zscore_array, model_number):
    print(f"model {model_number}")
    train_X = zscore_array[(group_arr != model_number) & (group_arr != 11)]
    test_X = zscore_array[group_arr == model_number]

    train_Y_index = class_arr[(group_arr != model_number) & (group_arr != 11)]
    test_Y_index = class_arr[(group_arr == model_number)]

    feature_count = train_X.shape[1]
    unique_classes = np.unique(train_Y_index)
    num_classes = len(unique_classes)

    # These arrays basically OHE the class to columns. Instead of a bunch of class numbers, we have an array with a
    # single `1` on each row indicating the class.
    train_Y = np.eye(num_classes)[train_Y_index]
    test_Y = np.eye(num_classes)[
        test_Y_index
    ]  # TODO: check that removing the -1 here fixed the indexing issue.

    es = EarlyStopping(
        monitor="loss", mode="min", verbose=2, patience=5, min_delta=0.02
    )

    val_model_path = str(
        (
            stored_models.get_model_dir(name)
            / f'model_files/val_{"{:02d}".format(model_number)}.hdf5'
        ).resolve()
    )

    mc = ModelCheckpoint(
        val_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    acc_model_path = str(
        (
            stored_models.get_model_dir(name)
            / f'model_files/acc_{"{:02d}".format(model_number)}.hdf5'
        ).resolve()
    )

    mc2 = ModelCheckpoint(
        acc_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_Y_index), y=train_Y_index
    )

    train_weights = dict(zip(range(num_classes), class_weights))

    model = Sequential()
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.add(Input(shape=(feature_count,)))
    model.add(
        Dense(
            feature_count,
            kernel_initializer="random_uniform",
            activation="relu",
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(
        train_X,
        train_Y,
        validation_data=(test_X, test_Y),
        epochs=120,
        batch_size=5000,
        verbose=2,
        class_weight=train_weights,
        callbacks=[es, mc, mc2],
    )

    test_Y_prediction_values = model.predict(test_X)
    test_Y_predicted = np.argmax(test_Y_prediction_values, axis=1)

    model_path = str(
        (
            stored_models.get_model_dir(name)
            / f'model_files/{"{:02d}".format(model_number)}.hdf5'
        ).resolve()
    )
    model.save(model_path)

    model_val = load_model(val_model_path)
    test_Y_prediction_values_val = model.predict(test_X)
    test_Y_predicted_val = np.argmax(test_Y_prediction_values_val, axis=1)

    model_acc = load_model(acc_model_path)
    test_Y_prediction_values_acc = model.predict(test_X)
    test_Y_predicted_acc = np.argmax(test_Y_prediction_values_acc, axis=1)

    K.clear_session()

    print(test_Y_predicted)


def initial_predict(model_name, zscore_array, group_arr, class_arr):
    test_X = zscore_array[group_arr == 11]
    test_Y_index = class_arr[group_arr == 11]

    y_hats = []

    stored_model_dir = stored_models.get_model_dir(model_name) / "model_files/"

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    print("Running initial model testing")
    for model_number in tqdm(range(1, 11)):
        model_full_name = f"{'{:02d}'.format(model_number)}.hdf5"
        model_path = stored_model_dir / model_full_name
        model = load_model(model_path)

        y_hat = model.predict(test_X, verbose=0)
        y_hats.append(y_hat)

        clear_session()
        del model
        del y_hat

    predicted_Y = np.sum(y_hats, axis=0)
    predicted_Y_index = np.argmax(predicted_Y, axis=1)

    return predicted_Y, predicted_Y_index
