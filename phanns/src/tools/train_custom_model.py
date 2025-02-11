import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from Bio import SeqIO
from plotly.io import to_html
from tqdm import tqdm

sys.path.append("..")  # TODO: Do I need to do this for pip installable w/o pyx files?

import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tools.model import SequentialNN
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from utils import calc, stored_models
from utils.data_handler import Data, fasta_count


def load_dataset(fasta_dir):
    if not Path(fasta_dir).exists():
        raise ValueError(f"Selected directory ({fasta_dir}) does not exist!")

    # collect files
    fastas = sorted(list(Path(fasta_dir).glob("*.fasta")))

    if len(fastas) == 0:
        raise ValueError(f"No .fasta files found at target directory: {fasta_dir}")

    # assign a numeric code to each base file name
    re_pattern = r"(?P<prefix>\d+)_(?P<name>.*)"
    group_names = set()
    for file in fastas:
        group_names.add(re.match(re_pattern, file.stem).group("name"))
    sorted_group_names = sorted(list(group_names))

    class_number_assignments = {x: i for i, x in enumerate(sorted_group_names)}

    for k, v in class_number_assignments.items():
        print(f"{k}:\t{v}")

    # init Data object
    num_proteins = fasta_count(fastas)
    data = Data(num_proteins)
    row_counter = 0

    # establish empty group and class arrays
    group_arr = np.zeros(num_proteins, dtype=int)
    class_arr = np.zeros(num_proteins, dtype=int)

    def process_sequence(record, data, row_counter, cls_number, group_number):
        sequence = record.seq.__str__().upper()
        row = data.feature_extract(sequence)
        data.add_to_array(row, row_counter, cls_number, group_number)

        group_arr[row_counter] = group_number
        class_arr[row_counter] = cls_number

    for file_path in fastas:
        match = re.search(re_pattern, file_path.stem)
        cls = match.group("name")
        group_number = match.group("prefix")
        cls_number = class_number_assignments[cls]

        print(f"{file_path}, class:{cls_number}, group:{group_number}")

        records = SeqIO.parse(file_path, "fasta")
        num_proteins_current_file = fasta_count([file_path])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            print("Building process pool")
            for _ in tqdm(range(num_proteins_current_file)):
                record = next(records)
                future = executor.submit(
                    process_sequence,
                    record,
                    data,
                    row_counter,
                    cls_number,
                    group_number,
                )
                futures.append(future)
                row_counter += 1

            print("Executing process pool")
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=num_proteins_current_file,
            ):
                future.result()

    print("Calculating z-score normalization")
    mean_array, stdev_array, zscore_array = calc.zscore(data.arr)

    zscore_array = np.array(zscore_array, dtype=np.float32)
    mean_array = np.array(mean_array, dtype=np.float32)
    stdev_array = np.array(stdev_array, dtype=np.float32)

    return (
        mean_array,
        stdev_array,
        zscore_array,
        group_arr,
        class_arr,
        sorted_group_names,
        data.arr,
    )


def train_new_pytorch_model(name, class_arr, group_arr, zscore_array, model_number):
    print(f"model {model_number}")

    # set up cuda
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # train/val/test split data
    train_X = zscore_array[(group_arr != model_number) & (group_arr != 11)]
    val_X = zscore_array[group_arr == model_number]

    train_Y_index = class_arr[(group_arr != model_number) & (group_arr != 11)]
    val_Y_index = class_arr[group_arr == model_number]

    unique_classes = sorted(np.unique(train_Y_index))
    num_classes = len(unique_classes)

    # define hyperparameters
    feature_count = train_X.shape[1]
    learning_rate = 0.000005
    epochs = 120
    batch_size = 5000
    best_val_loss = float("inf")
    best_val_accuracy = 0
    patience = 10
    break_in = 15
    min_delta = 0.02
    patience_counter = 0
    model_path = str(
        (
            stored_models.get_model_dir(name)
            / f'model_files/{"{:02d}".format(model_number)}.pt'
        ).resolve()
    )
    print(
        "\t".join(
            [
                f"Learning rate: {learning_rate}",
                f"patience: {patience}",
                f"break_in: {break_in}",
                f"min_delta: {min_delta}",
            ]
        )
    )

    # class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_Y_index), y=train_Y_index
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    # create the model
    model = SequentialNN(feature_count, num_classes).to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08
    )

    # create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_X), torch.LongTensor(train_Y_index)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(val_X), torch.LongTensor(val_Y_index))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model_training_scores = defaultdict(list)

    # training loop
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}\tpatience={patience_counter}")
        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )
        model_training_scores["val_acc"].append(val_accuracy)
        model_training_scores["val_loss"].append(val_loss)
        model_training_scores["train_acc"].append(train_accuracy)
        model_training_scores["train_loss"].append(train_loss)

        # Save best accuracy model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Model saved (best validation accuracy: {val_accuracy})")

        # Early stopping
        if val_loss <= best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), val_model_path)
            # print("Model saved (best validation loss)")
        else:
            if epoch >= break_in:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping\n")
                    break

    return feature_count, num_classes, model_training_scores


def plot_training_loss_acc(model_name, model_training_scores_dict: dict):
    file_path = Path(f"{model_name}_train_acc_loss.html")
    fig = go.Figure()
    for model, model_values in model_training_scores_dict.items():
        train_acc = model_values["train_acc"]
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(len(train_acc))],
                y=train_acc,
                name=f"model {model}-train accuracy",
                mode="lines",
                line=dict(color="royalblue"),
            )
        )
        train_loss = model_values["train_loss"]
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(len(train_loss))],
                y=train_loss,
                name=f"model {model}-train loss",
                mode="lines",
                line=dict(color="royalblue", dash="dash"),
            )
        )
        val_acc = model_values["val_acc"]
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(len(val_acc))],
                y=val_acc,
                name=f"model {model}-validation accuracy",
                mode="lines",
                line=dict(color="firebrick"),
            )
        )
        val_loss = model_values["val_loss"]
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(len(val_loss))],
                y=val_loss,
                name=f"model {model}-validation loss",
                mode="lines",
                line=dict(color="firebrick", dash="dash"),
            )
        )
    fig.update_layout(
        height=800,
        width=800,
        title="Train/Validation Acc/Loss per epoch, per model",
        xaxis_title="Epoch",
        yaxis_title="value",
    )

    with open(file_path, "w") as output:
        output.write(to_html(fig, include_plotlyjs="cdn"))

    print(f"Accuracy and losses graph written to {Path(file_path).absolute()}")
