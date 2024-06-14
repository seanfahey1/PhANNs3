import argparse
import gc
import pickle as p
import sys
import time

from tools import predict, train_custom_model

# fmt: off
gc.disable()
sys.path.append("..")
from utils.stored_models import (export_model, list_models, load_model,
                                 load_stored_model, move_model, remove_model,
                                 store_newly_generated_model, validate_model)

# fmt: on


def get_export_args():
    parser = argparse.ArgumentParser(
        description="Export a pre-stored model. Warning: These files can be VERY large (often > 30Gb)."
    )
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        help="Name of the stored model to be exported.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        default=".",
        help="Optional directory where the tar file should be written. (default: %(default)s)",
    )
    args, _ = parser.parse_known_args()
    return args


def get_load_args():
    parser = argparse.ArgumentParser(description="Store a pre-trained model.")
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        help="Name of the stored model to be exported.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="Path to the .tar.gz model file to load.",
    )
    args, _ = parser.parse_known_args()
    return args


def get_rm_model_args():
    parser = argparse.ArgumentParser(description="Delete a pre-stored model.")
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        help="Name of the stored model to be deleted.",
    )
    args, _ = parser.parse_known_args()
    return args


def get_move_model_args():
    parser = argparse.ArgumentParser(description="Delete a pre-stored model.")
    parser.add_argument(
        "-o",
        "--old_model_name",
        required=True,
        help="Name of the stored model to be moved.",
    )
    parser.add_argument(
        "-n",
        "--new_model_name",
        required=True,
        help="New name for the stored model.",
    )
    args, _ = parser.parse_known_args()
    return args


def get_train_args():
    parser = argparse.ArgumentParser(
        description="Tests a PhANNs model using pre-loaded test data. Must run load.py step first."
    )
    parser.add_argument(
        "-f",
        "--fasta_dir",
        required=True,
        help="Path to the directory storing the de-de-replicated (pre-split) fasta files.",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        help="Name of the stored model.",
    )
    args, _ = parser.parse_known_args()
    return args


def get_classify_args():
    parser = argparse.ArgumentParser(
        description="Tests a PhANNs model using pre-loaded test data. Must run load.py step first."
    )
    parser.add_argument(
        "-f",
        "--fasta",
        required=True,
        help="Path to the target fasta file.",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        help="Name of the stored model.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        default="./output.csv",
        help="Optional relative path of the output csv file. (default: %(default)s)",
    )
    args, _ = parser.parse_known_args()
    return args


def rm():
    rm_args = get_rm_model_args()
    print("Starting model removal step.")
    remove_model(rm_args.model_name)


def move():
    move_args = get_move_model_args()
    print("Starting model moving step.")
    move_model(move_args.old_model_name, move_args.new_model_name)


def export():
    export_args = get_export_args()
    print("Starting model export step.")
    export_model(export_args.model_name, export_args.output_file)


def load():
    load_args = get_load_args()
    print("Starting model loading step.")
    load_model(load_args.model_name, load_args.input_file)


def train():
    gc.enable()
    train_args = get_train_args()

    # print("Starting data loading step.")
    # (
    #     mean_array,
    #     stdev_array,
    #     zscore_array,
    #     group_arr,
    #     class_arr,
    #     sorted_group_names,
    #     data_array,
    # ) = train_custom_model.load_dataset(train_args.fasta_dir)

    # print("Storing newly generated data.")
    # store_newly_generated_model(
    #     train_args.model_name,
    #     stdev_array,
    #     mean_array,
    #     group_arr,
    #     class_arr,
    #     sorted_group_names,
    # )

    # print("Writing caches.")
    # with open("mean.cache", "wb") as m:
    #     p.dump(mean_array, m)
    # with open("stdev.cache", "wb") as m:
    #     p.dump(stdev_array, m)
    # with open("zscore.cache", "wb") as m:
    #     p.dump(zscore_array, m)
    # with open("group.cache", "wb") as m:
    #     p.dump(group_arr, m)
    # with open("class.cache", "wb") as m:
    #     p.dump(class_arr, m)
    # with open("sorted_groups.cache", "wb") as m:
    #     p.dump(sorted_group_names, m)
    # with open("raw_data.cache", "wb") as m:
    #     p.dump(data_array, m)

    print("Loading from cache.")
    with open("mean.cache", "rb") as m:
        mean_array = p.load(m)
    with open("stdev.cache", "rb") as m:
        stdev_array = p.load(m)
    with open("zscore.cache", "rb") as m:
        zscore_array = p.load(m)
    with open("group.cache", "rb") as m:
        group_arr = p.load(m)
    with open("class.cache", "rb") as m:
        class_arr = p.load(m)
    with open("sorted_groups.cache", "rb") as m:
        sorted_group_names = p.load(m)
    with open("raw_data.cache", "rb") as m:
        data_array = p.load(m)

    print("Starting model training step.")
    for model_number in range(1, 11):
        train_custom_model.train_new_model(
            train_args.model_name, class_arr, group_arr, zscore_array, model_number
        )
        time.sleep(2)
        gc.collect()

    # this may have fixed everything...
    test_X = zscore_array[group_arr == 11]
    test_y = class_arr[group_arr == 11]

    with open("test_X.cache", "wb") as m:
        p.dump(test_X, m)
    with open("test_y.cache", "wb") as m:
        p.dump(test_y, m)

    predicted_Y, predicted_Y_index = predict.predict(
        train_args.model_name, test_X=test_X
    )

    class_number_assignments = {i: x for i, x in enumerate(sorted_group_names)}
    predicted_class = [class_number_assignments[x] for x in predicted_Y_index]
    true_class = [class_number_assignments[x] for x in test_y]

    print("Starting initial prediction step.")
    predict.write_initial_prediction_outputs(
        f"{train_args.model_name}_initial_results.csv",
        predicted_Y,
        predicted_class,
        sorted_group_names,
        true_class,
    )

    predict.confusion_matrix(
        f"{train_args.model_name}_initial_confusion_matrix.html",
        true_class,
        predicted_class,
    )
    gc.disable()


def classify():
    classify_args = get_classify_args()
    if not validate_model(classify_args.model_name):
        sys.exit(
            "Selected model is invalid. Try running `phanns list_models` to check available model."
        )

    print("Starting prediction step")
    _, mean_arr, std_arr, sorted_group_names = load_stored_model(
        classify_args.model_name
    )

    zscore_array, fasta_headers = predict.load_dataset(
        classify_args.fasta, mean_arr, std_arr
    )
    prediction_scores, prediction = predict.predict(
        classify_args.model_name, zscore_array
    )

    class_number_assignments = {i: x for i, x in enumerate(sorted_group_names)}

    predicted_class = [class_number_assignments[x] for x in prediction]
    predict.write_prediction_outputs(
        classify_args.output_file,
        prediction_scores,
        predicted_class,
        fasta_headers,
        sorted_group_names,
    )


def main():
    if len(sys.argv) < 2:
        print(
            """
Welcome to PhANNs. Please select a PhANNs utility to execute.
Options:

    `phanns list_models` to view a list of available models
    `phanns train` to train a new PhANNs model from a prepared dataset
    `phanns load` to save a pre-trained PhANNs model (.tar file) for later use (not yet working)
    `phanns classify` to classify proteins in a fasta file using a pre-loaded PhANNs model
    `phanns export` to export a model as a .tar.gz file
    `phanns rm` to delete a pre-saved model
    """
        )
        sys.exit()

    else:
        try:
            assert sys.argv[1] in [
                "list_models",
                "train",
                "load",
                "classify",
                "export",
                "rm",
                "move",
            ]
        except AssertionError:
            raise AttributeError(f"{sys.argv[1]} is not a valid command.")

        globals()[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
