# PhANNs

[PhANNs](https://github.com/Adrian-Cantu/PhANNs) is a tool to classify the protein class of any phage ORF. It uses an ensemble of Artificial Neural Networks.

This project is PhANNs 3.1. PhANNs 3.1 is a major revision of the PhANNs Neural Network with a focus on improved accuracy, utility, and speed. Like the original PhANNs program, PhANNs 3.1 is a 10-fold cross validated linear feed-forward Neural Network that utilizes the same overall architecture, allowing for direct comparisons between the two versions. The new PhANNs 3.1 program features new phage structural protein classes and an expanded, more up-to-date, and more heavily curated training dataset.

# Table of Contents
- [PhANNs](#phanns)
- [Table of Contents](#table-of-contents)
- [How to Use PhANNs](#how-to-use-phanns)
  - [Using a pre-trained model](#using-a-pre-trained-model)
    - [Pre-conditions](#pre-conditions)
    - [Background information](#background-information)
    - [Steps to classify a fasta-formatted file containing target ORFs](#steps-to-classify-a-fasta-formatted-file-containing-target-orfs)
  - [Training your own model](#training-your-own-model)
    - [Steps to train a model](#steps-to-train-a-model)
- [Paper](#paper)
- [How to cite this project](#how-to-cite-this-project)


# How to Use PhANNs

## Using a pre-trained model

Pre-trained models are too large to be included in this repo. For now they are available by reaching out to me directly (seanfahey21@gmail.com) or Dr. Anca Segall (asegall@sdsu.edu).

### Pre-conditions

1. A `.fasta` formatted file containing target ORFs to classify.
2. A *nix based system with a CUDA enabled GPU (cuda v12.5 or later) and at least 16 Gb VRAM.
3. python 3.8 or later.
   1. Check with `python3 --version`
4. Clone this repo.
5. `cd PhANNs3`
6. Install the dependencies in the requirements file to a virtual environment.
   1. `python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt`
7. If no phanns model is available, load a model:
   1. `cd phanns/src`
   2. `python3 phanns.py load -n <desired_model_name> -i <path/to/input/file.tar.gz>`

### Background information

Available phanns commands are available by calling the `phanns.py` file from its directory with no arguments:

```bash
cd phanns/src

python3 phanns.py

Welcome to PhANNs. Please select a PhANNs utility to execute.
Options:

    `phanns list_models` to view a list of available models
    `phanns train` to train a new PhANNs model from a prepared dataset
    `phanns load` to save a pre-trained PhANNs model (.tar file) for later use (not yet working)
    `phanns classify` to classify proteins in a fasta file using a pre-loaded PhANNs model
    `phanns export` to export a model as a .tar.gz file
    `phanns rm` to delete a pre-saved model
```

Optionally you can run the phanns command followed by `-h` to see a list of required arguments for that command.

```bash
python phanns.py classify -h
usage: phanns.py [-h] -f FASTA -n MODEL_NAME [-o OUTPUT_FILE]

Tests a PhANNs model using pre-loaded test data. Must run load.py step first.

options:
  -h, --help            show this help message and exit
  -f FASTA, --fasta FASTA
                        Path to the target fasta file.
  -n MODEL_NAME, --model_name MODEL_NAME
                        Name of the stored model.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Optional relative path of the output csv file. (default: ./output.csv)
```

### Steps to classify a fasta-formatted file containing target ORFs

1. Navigate to the directory containing the `phanns.py` file:
   1. `cd phanns/src`
2. Run the classification command:
   1. `phanns classify -f <path/to/fasta/file> -n <model_name> [-o <output/path> optional]`

## Training your own model

You can also training a custom PhANNs model by providing your own source dataset. This dataset should consist of fasta formatted files for each "class". Perform a 11-fold split of each of these files. The resulting names should be `1_classname.fasta` through `11_classname.fasta`. The first 10 of these files will be used for training and validation. File #11 will be automatically set aside and used for testing the model at the end of the training process. Leave file #11 blank to skip this step and produce a model without retaining test data. [See this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007845) for an example of how the split process works in the published versions of PhANNs. A comprehensive non-random split process is essential to ensure proper training.

### Steps to train a model

1. Move the 11-fold split fasta-formatted files to a directory containing no other `.fasta` files.
2. Run the train command
   1. `phanns train -f <path/to/dir/containing/fasta/files/> -n <model_name>`
3. Note: This command may take a long time to run. If `nohup` is installed, you can optionally prefix the command with nohup to prevent tty disconnects from interrupting the process
   1. `nohup phanns train -f <path/to/dir/containing/fasta/files/> -n <model_name> > training_stdout_log.txt`
   2. `tail -f training_stdout_log.txt`

# Paper

PhANNs paper is available [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007845).


# How to cite this project

Please cite as
```
Cantu, V.A., Salamon, P., Seguritan, V., Redfield, J., Salamon, D., Edwards, R.A., and Segall, A.M. (2020). PhANNs, a fast and accurate tool and web server to classify phage structural proteins. PLOS Computational Biology 16, e1007845. 10.1371/journal.pcbi.1007845.
```
