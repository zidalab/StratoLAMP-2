"""
Step 2: Precipitate Detection on Cropped Droplet Images

This script:
1. Loads a pretrained YOLOv11 segmentation model
2. Runs segmentation on each cropped droplet image
3. Saves visualized prediction results and YOLO-format mask labels

python precipitate_detection.py \
  --root_dir /path/to/data \
  --crop_dirs crop \
  --segmentation_model /path/to/precipitate_detection.pt
"""

import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

ROOT = "/path/to/data"

# Input folders containing cropped droplet images
CROP_DIR_LIST = ['crop']

# Segmentation model path
SEGMENT_MODEL_PATH = "/path/to/segmentation_model.pt"


def save_segmentation_masks(results, image_name, label_dir):
    """
    Save segmentation masks in YOLO polygon format.

    Args:
        results (list): List of YOLO Results
        image_name (str): Image filename
        label_dir (str): Output directory for YOLO .txt label files
    """
    txt_path = os.path.join(label_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_path, 'w') as f:
        for result in results:
            if result.masks is not None:
                for mask, cls in zip(result.masks.xy, result.boxes.cls):
                    norm_mask = mask / np.array([result.orig_shape[1], result.orig_shape[0]])
                    polygon = [f"{x:.6f} {y:.6f}" for x, y in norm_mask]
                    f.write(f"{int(cls)} {' '.join(polygon)}\n")


def run_precipitate_detection():
    """
    Main loop to process all cropped images and save results.
    """
    model = YOLO(SEGMENT_MODEL_PATH)
    model.eval()

    for folder in CROP_DIR_LIST:
        input_dir = os.path.join(ROOT, folder)
        output_dir = os.path.join(ROOT, f"{folder}_precipitate_detection")
        label_dir = os.path.join(output_dir, "labels")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        print(f"Processing: {input_dir}")

        for img_name in os.listdir(input_dir):
            if not img_name.lower().endswith('.png'):
                continue

            img_path = os.path.join(input_dir, img_name)

            results = model.predict(img_path, iou=0.2, conf=0.25)

            for result in results:
                # Save visualization
                vis_img = result.plot(line_width=1, font_size=0.1)[:, :, ::-1]  # Convert RGB to BGR
                cv2.imwrite(os.path.join(output_dir, img_name), vis_img)

            # Save label file
            save_segmentation_masks(results, img_name, label_dir)

        print(f"[✓] Results saved to: {output_dir}")
        print(f"[✓] Labels saved to: {label_dir}")


if __name__ == "__main__":
    run_precipitate_detection()