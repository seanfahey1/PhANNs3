import argparse
import sys

from tools import predict, train_custom_model

# fmt: off
sys.path.append("..")
from utils.stored_models import (export_model, list_models, load_model,
                                 load_stored_model, remove_model,
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


def export():
    export_args = get_export_args()
    print("Starting model export step.")
    export_model(export_args.model_name, export_args.output_file)


def load():
    load_args = get_load_args()
    print("Starting model loading step.")
    load_model(load_args.model_name, load_args.input_file)


def train():
    train_args = get_train_args()
    print("Starting model training step.")
    (
        mean_array,
        stdev_array,
        zscore_array,
        group_arr,
        class_arr,
        sorted_group_names,
    ) = train_custom_model.load_dataset(train_args.fasta_dir)
    store_newly_generated_model(
        train_args.model_name, stdev_array, mean_array, sorted_group_names
    )

    train_custom_model.train_new_model(
        train_args.model_name, class_arr, group_arr, zscore_array
    )
    predicted_Y, predicted_Y_index = predict.predict(
        train_args.model_name, zscore_array=zscore_array[group_arr == 11]
    )

    class_number_assignments = {i: x for i, x in enumerate(sorted_group_names)}
    predicted_class = [class_number_assignments[x] for x in predicted_Y_index]

    predict.write_initial_prediction_outputs(
        f"{train_args.model_name}_initial_results.csv",
        predicted_Y,
        predicted_class,
        sorted_group_names,
        class_arr,
    )


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
            ]
        except AssertionError:
            raise AttributeError(f"{sys.argv[1]} is not a valid command.")

        globals()[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
