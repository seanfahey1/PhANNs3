# PhANNs 3 — Special Setup Guide (mac/*nix)

This guide walks through a clean setup from cloning the repository to running PhANNs commands.

## 1) Prerequisites

- macOS or Linux shell environment
- `git`
- Python `3.8+` (recommended: Python `3.10+`)
- `pip`
- (Recommended on macOS) Xcode Command Line Tools for native builds:

```bash
xcode-select --install
```

### macOS: install Python with Homebrew + pyenv

First, check the currently installed python version:

```bash
python3 --version
```

Follow these commands only if `python3` is missing or older than `3.8`. These commands 
only need to be run ONE time. After completing this section, python will be permanently
installed on your system.

1. Check whether Homebrew is installed:

```bash
command -v brew
```

If that prints nothing, install Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install `pyenv`:

```bash
brew update
brew install pyenv
```

3. Add `pyenv` to your shell startup file:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc
source ~/.zshrc
```

5. Install and select a Python version for this machine:

```bash
pyenv install 3.11.11
pyenv global 3.11.11
python3 --version
```

### GPU note

PhANNs workflows are designed around CUDA-enabled GPUs for model training/inference performance. If you plan to train models or run large inference jobs, use a machine with NVIDIA CUDA support. Non-CUDA systems will need a small modification in the code to suppress possible warnings/errors,

---

## 2) Clone the repository

```bash
git clone https://github.com/seanfahey1/PhANNs3.git
cd PhANNs3
```

---

## 3) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install uv
```

You should now see `(.venv)` in your shell prompt.

---

## 4) Install Python dependencies

From the repository root:

```bash
uv pip install -r requirements.txt
```

Optional (developer-style install of the local package metadata):

```bash
uv pip install -e .
```

---

## 5) Verify the CLI entry script

PhANNs commands are exposed through `phanns/src/phanns.py`.

```bash
cd phanns/src
python3 phanns.py
```

This should print the available commands such as:

- `list_models`
- `train`
- `load`
- `classify`
- `export`
- `rm`

For command-specific help, append `-h`, for example:

```bash
python3 phanns.py classify -h
```

---

## 6) Run examples

### Pre-conditions

Move to the target directory containing the phanns.py file. This command assumes that
it is run in the same directory that `git clone` was run in:

```bash
cd PhANNs3/phanns/src
```

#### A) Load a pre-trained model tarball

```bash
python3 phanns.py load -n my_model -i /path/to/model.tar.gz
```

#### B) Classify proteins from a FASTA file

```bash
python3 phanns.py classify -f /path/to/targets.fasta -n my_model -o ./output.csv
```

#### C) Train a custom model from an pre-split data directory

```bash
python3 phanns.py train -f /path/to/split_fasta_dir -n my_custom_model
```

Dataset expectation: files named `1_classname.fasta` through `11_classname.fasta` for each class.

---

## 7) Common troubleshooting

- **`python3: command not found`**
  - Install Python 3 and re-open your shell.

- **Build/compile errors while installing dependencies**
  - Ensure Xcode Command Line Tools are installed on macOS (`xcode-select --install`).
  - Upgrade pip: `python3 -m pip install --upgrade pip`.

- **CUDA/GPU errors**
  - Confirm your runtime has compatible NVIDIA drivers/CUDA libraries.
  - If no CUDA GPU is available, training/inference performance and compatibility may be limited.

- **Model not found during classify**
  - Run `python3 phanns.py list_models` and verify the `-n` model name matches.

---

## 8) Deactivate environment when done

```bash
deactivate
```
