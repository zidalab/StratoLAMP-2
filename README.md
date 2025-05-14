# 🧪 StratoLAMP-2
## 📦 Overview
This project is specifically tailored for StratoLAMP-2, a method designed for dual nucleic acid quantification by tracking and classifying polydisperse droplets via flow-based analysis. The classification is based on the stratification of precipitates within each droplet. The [YOLOv11](https://github.com/ultralytics/ultralytics) has been customized to meet our specific requirements. This repository contains the modified YOLOv11 model, including training code, pre-trained weights, tracking and instance segmentation detection code, precipitate segmentation code, multi-target quantification code, model configuration files, and other necessary resources.

### ✅ This repo includes:

- `tracking&quantification.py`: Main pipeline to process videos, crop and classify droplets, and quantify target concentrations.
- `train_yolov11.py`: Unified script to train YOLOv11 models for segmentation and classification tasks.
- `precipitate_detection.py`: Main pipeline for single-droplet precipitate detection, generating binary masks of precipitate regions.
- `quantification.py`: Implements Newton-Raphson-based digital quantification and droplet volume calculation.
- `*.yaml`: Configuration files for dataset paths, tracker (BoT-SORT), and YOLOv11 architecture.
- `environment.yml`: Reproducible Conda environment for all dependencies.

---

## 🧬 Principle

This pipeline was designed for digital droplet LAMP systems using **bright-field**, where:
- Droplets vary in size (**polydisperse**).
- Amplification products generate **precipitates** that serve as detection signals (replacing fluorescence readout)
- Droplets containing different targets generate **distinguishable precipitate levels**
- Dual-target concentration model:
For duplex detection, droplets can be categorized into four types: \
(1) empty, \
(2) containing only Target 1, \
(3) containing only Target 2, \
(4) containing both targets (denoted as Target 1&2). \
The concentrations of each target,  $c_1$ and $c_2$, can be calculated by solving the corresponding equations.

$$
v_{\text{Total}} = \sum_{i = 1}^{N_1} \frac{v_{1,i}}{\left[1 - \exp(-v_i \cdot c_1)\right]} + \sum_{i = 1}^{N_3} \frac{v_{3,i}}{\left[1 - \exp(-v_i \cdot c_1)\right]}
$$

$$
v_{\text{Total}} = \sum_{i = 1}^{N_2} \frac{v_{2,i}}{\left[1 - \exp(-v_i \cdot c_2)\right]} + \sum_{i = 1}^{N_3} \frac{v_{3,i}}{\left[1 - \exp(-v_i \cdot c_2)\right]}
$$

<!-- ![Dual-target Quantification Equations](/docs/dual_target_equation.png) -->  

Where $v_{\text{Total}}$ presents the total volume of the droplets, $N_1, N_2$, and $N_3$ represent the counts of droplets containing only Target 1, only Target 2, and Target 1&2, respectively.  $v_{1,i}, v_{2,i}$, and $v_{3,i}$ represent the volume of the $i$-th droplets the group containing only Target 1, only Target 2, and Target 1&2, respectively. 

---

## 🧰 Usage

### 🐍 Create Conda Environment(Windows 11)
We recommend using Anaconda on Windows to manage dependencies and run the project in a dedicated virtual environment. 

Step 1: Open Anaconda Prompt and create the environment
```bash
conda env create -f environment.yml
```
Step 2: Activate the environment
```bash
conda activate Yolov11
```
Step 3: Verify installation

You should now be able to run python, ultralytics, and torch commands inside this environment.
For alternative installation methods, including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and building from source via Git, please consult the [Quickstart Guide](https://docs.ultralytics.com/quickstart/).

### 🖐️ Train Custom Models
You can use the `train_yolov11.py` script to train different YOLOv11 models for segmentation or classification tasks. It automatically loads the appropriate configuration and launches training.

```bash
python train_yolov11.py droplets
```
- Trains a segmentation model for droplet 
- Uses dataset defined in ./datasets/droplets_segmentation.yaml
- Default image size: 1280 px; epochs: 100

```bash
python train_yolov11.py precipitate
```
- Trains a segmentation model to identify precipitate regions within droplets
- Uses dataset ./datasets/precipitate_detection.yaml
- Default batch size: 32; learning rate: 1e-6; epochs: 300

```bash
python train_yolov11.py mnist
```
- Trains a classifier to distinguish droplet types (e.g., empty/small/medium/large)
- Uses mnist160-formatted image folder with 4 classes (0–3)
- Default image size: 128 px; batch: 32

Configuration parameters like epochs, batch size, learning rate, and image size are specified in the script and can be customized by modifying the config dataclasses.

###  ▶️ Run the Main Pipeline
```bash
python tracking_pineline.py \
    --root_dir "/path/to/your/data" \
    --video_labels
    --detection_model "/path/to/segmentation_model.pt" \
    --classification_model "/path/to/classification_model.pt"
```
This performs tracking, cropping, classification, and quantification.

Output includes:

- `_detected.avi` with segmentation overlays

- `_tracking.xlsx` per-droplet class and area

- `_quantification.xlsx` with concentrations of target1 and target2

- `_consolidated_results.xlsx` for all videos

```bash
python precipitate_detection.py \
  --root_dir ./data \
  --crop_dirs 1_crop 2_crop \
  --segmentation_model ./weights/yolo11n-seg.pt
```
This script performs **segmentation of precipitate regions** in previously cropped droplet images.

Output includes::

- `*_precipitate_detection/` folders containing visualized prediction results (overlaid `.png` images)
- `*_precipitate_detection/labels/` containing YOLO-format polygon mask labels (`.txt`)
---

## 📄 License Notice

This project uses Ultralytics YOLOv11 under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

If you use this repository in academic or commercial contexts, please cite Ultralytics as:


```bash
@software{yolov11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLOv11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```


Our paper describing this work is currently under review. Citation details will be provided here once the paper is published.
