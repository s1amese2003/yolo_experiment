# Server training checklist

This project has already prepared the dataset locally under:

```text
datasets/remote_dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Dataset summary:

```text
train: 798 images
val:   228 images
test:  115 images
total: 1141 images
```

Classes in `configs/remote.yaml`:

```text
0 aircraft
1 oiltank
2 overpass
3 playground
```

The dataset contains 34 background images with empty label files, all from
`playground`. YOLO can train with empty label files.

## 1. Upload files to the server

Upload the project code and the prepared dataset directory. If you use Git,
remember that `.gitignore` ignores `datasets/`, so the dataset must be copied
separately.

Recommended upload content:

```text
remote_yolo_experiment/
  configs/
  src/
  datasets/remote_dataset/
  requirements.txt
  run_train.ps1
  README.md
```

The raw zip files and `datasets/raw/source` are not required on the server once
`datasets/remote_dataset` has been generated.

## 2. Create environment

Recommended server baseline:

```text
OS: Ubuntu 20.04/22.04 or a similar Linux image
Python: 3.10-3.12 recommended
GPU: NVIDIA GPU with at least 8 GB VRAM; 12 GB+ is more comfortable
Disk: 15 GB+ free for dataset, weights, caches, and training outputs
RAM: 16 GB+ recommended
Network: needed for pip packages and first-time YOLO weight downloads
```

```bash
cd remote_yolo_experiment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Check GPU:

```bash
nvidia-smi
```

Install the CUDA-enabled PyTorch build that matches the server driver/CUDA
runtime. Use the selector at https://pytorch.org/get-started/locally/ and then
install the project requirements. Example for a CUDA 12.6-compatible server:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If OpenCV reports `libGL.so.1` or `libglib` errors on a minimal Linux image:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

If the server cannot access the internet, upload these files manually to the
project root before training:

```text
yolo11n.pt
yolo11s.pt
```

Then change `configs/experiments.yaml` model names to local paths if needed,
for example `./yolo11n.pt`.

## 2.1 Pre-flight checks

Run these checks before training:

```bash
pwd
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
test -d datasets/remote_dataset/images/train
test -d datasets/remote_dataset/labels/train
find datasets/remote_dataset/images/train -type f | wc -l
find datasets/remote_dataset/images/val -type f | wc -l
find datasets/remote_dataset/images/test -type f | wc -l
```

Expected dataset counts:

```text
train images: 798
val images:   228
test images:  115
```

## 3. Smoke test

Run a quick test before the full experiment:

```bash
python src/train_experiments.py --device 0 --quick
```

## 4. Full training

```bash
python src/train_experiments.py --device 0
```

Training outputs are written to:

```text
runs/remote_yolo/
```

## 5. Generate thesis tables

After training finishes:

```bash
python src/benchmark_speed.py --split val --samples 100 --device 0
python src/summarize_results.py
```

The main table files are:

```text
reports/speed_results.csv
reports/thesis_results.csv
reports/thesis_results.md
```
