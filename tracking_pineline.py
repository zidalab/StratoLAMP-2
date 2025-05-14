"""
Video Tracking and Droplet Analysis Pipeline

This script performs:
1. Video frame processing with YOLO object detection and tracking
2. Droplet cropping, classification, and analysis
3. Quantitative analysis of nucleic acid concentrations

Usage examples:
  python tracking_pineline.py \
    --root_dir "/path/to/your/data" \
    --video_labels
    --detection_model "/path/to/segmentation_model.pt" \
    --classification_model "/path/to/classification_model.pt"

"""

import os
import time
import math
import cv2
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from ultralytics import YOLO
from units.quantification import calculate_volume, get_concentration

# Class name mapping
CLASS_NAME_MAP = {
    '1': 'empty',
    '2': 'small',
    '3': 'medium',
    '4': 'large',
}

# Global constants
SCALE_BAR = 1.72  # um/px
D_CR = 95  # Critical diameter
CROP_SIZE = 128  # Size for cropped images
MAX_SAVE_PER_ID = 5  # Max images to save per track ID
MIN_CIRCULARITY = 0.75  # Minimum circularity for valid droplets
CONFIDENCE_THRESHOLD = 0.7  # Minimum classification confidence


def crop_with_mask(frame_count, results, image, save_dir, wh=CROP_SIZE,
                   label=None, classifier=None, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Process detected droplets by cropping, sharpening, and classifying them.

    Args:
        frame_count (int): Current frame number
        results (YOLO.Results): Detection results from YOLO model
        image (np.array): Current frame image
        save_dir (str): Directory to save cropped images
        wh (int): Size for resizing cropped images
        label (str): Video label for classification
        classifier (YOLO): Classification model
        confidence_threshold (float): Minimum confidence for valid classification

    Returns:
        None (saves processed images and updates track_id_info)
    """
    img_height, img_width = image.shape[:2]

    for mask, box, track_id in zip(results[0].masks.data,
                                   results[0].boxes.xyxyn,
                                   results[0].boxes.id):
        mask = mask.cpu().numpy().astype(np.uint8)
        track_id = int(track_id)

        # Skip if we've saved enough images for this track_id
        if track_id in track_id_counter and track_id_counter[track_id] >= MAX_SAVE_PER_ID:
            continue

        # Apply mask to isolate droplet
        mask_rgb = cv2.merge((mask, mask, mask)) / 255
        masked_image = image * mask_rgb
        masked_image = (masked_image * 255).astype(np.uint8)

        # Find contours and validate droplet shape
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)

        # Calculate circularity to filter partial droplets
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < MIN_CIRCULARITY:
            continue

        # Create mask from largest contour
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(masked_image, masked_image, mask=contour_mask)

        # Calculate equivalent diameter and filter by size
        area_um2 = area * SCALE_BAR * SCALE_BAR
        D_eq = 2 * np.sqrt(area_um2 / np.pi)
        if D_eq > 180:  # Skip overly large droplets
            continue

        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.cpu().numpy()
        x1 = int(x1 * img_width)
        y1 = int(y1 * img_height)
        x2 = int(x2 * img_width)
        y2 = int(y2 * img_height)

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, min(x1, img_width)), max(0, min(y1, img_height))
        x2, y2 = max(0, min(x2, img_width)), max(0, min(y2, img_height))

        # Crop and process the droplet image
        cropped_image = masked_image[y1:y2, x1:x2]
        white_bg = np.ones_like(cropped_image) * 255
        mask = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Combine droplet with white background
        masked_crop = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        white_bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
        final_crop = cv2.add(masked_crop, white_bg)

        # Resize and sharpen
        resized = cv2.resize(final_crop, (wh, wh), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(resized, -1, kernel)

        # Classify droplet if classifier is provided
        if classifier:

            # Get classification results
            with torch.set_grad_enabled(True):
                class_result = classifier(sharpened)

            class_id = class_result[0].probs.top1
            confidence = class_result[0].probs.data.tolist()

            # Skip low confidence predictions
            if confidence[class_id] < confidence_threshold:
                continue

            # Update tracking info
            track_id_info[track_id]['classes'].append(int(class_id))
            track_id_info[track_id]['confidences'].append(confidence)
            track_id_info[track_id]['areas'].append(area)


            if track_id not in track_id_counter:
                track_id_counter[track_id] = 0


        # Save processed droplet image
        save_path = os.path.join(save_dir, f"droplet_{track_id}_{track_id_counter[track_id]}.png")
        cv2.imwrite(save_path, sharpened)
        track_id_counter[track_id] += 1


def apply_clahe(frame):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        frame (np.array): Input image frame

    Returns:
        np.array: Contrast-enhanced grayscale image
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def calculate_bounds(df):
    """
    Calculate lower and upper bounds for droplet areas using IQR method.

    Args:
        df (pd.DataFrame): DataFrame containing droplet areas

    Returns:
        tuple: (lower_bound, upper_bound) for valid droplet areas
    """
    Q1 = df['area'].quantile(0.25)
    Q3 = df['area'].quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def quantify_targets(df, root_dir, label, consolidated_df=None):
    """
    Perform quantitative analysis of nucleic acid concentrations.

    Args:
        df (pd.DataFrame): Droplet classification and area data
        root_dir (str): Root directory for saving results
        label (str): Video label identifier
        consolidated_df (pd.DataFrame): Existing consolidated results

    Returns:
        pd.DataFrame: Updated consolidated results with new quantification data
    """
    save_path = os.path.join(root_dir, f"{label}_quantification.xlsx")

    # Filter outliers based on area
    lower, upper = calculate_bounds(df)
    filtered_df = df[(df['area'] >= lower) & (df['area'] <= upper)]

    # Group areas by classification
    class_areas = {
        0: filtered_df[filtered_df['class'] == 0]['area'].tolist(),
        1: filtered_df[filtered_df['class'] == 1]['area'].tolist(),
        2: filtered_df[filtered_df['class'] == 2]['area'].tolist(),
        3: filtered_df[filtered_df['class'] == 3]['area'].tolist()
    }

    # Convert areas to volumes
    class_volumes = {
        cls: [calculate_volume(area, SCALE_BAR, D_CR) for area in areas]
        for cls, areas in class_areas.items()
    }

    # Calculate statistics
    total_droplets = sum(len(v) for v in class_volumes.values())
    avg_volume = sum(sum(vols) for vols in class_volumes.values()) / total_droplets
    avg_radius = ((3 * avg_volume * 1E9) / (4 * math.pi)) ** (1 / 3)

    # Target 1 quantification (classes 1 & 3 vs 0 & 2)
    pos_t1 = class_volumes[1] + class_volumes[3]
    neg_t1 = class_volumes[0] + class_volumes[2]
    conc_t1 = get_concentration(pos_t1, neg_t1)

    # Target 2 quantification (classes 2 & 3 vs 0 & 1)
    pos_t2 = class_volumes[2] + class_volumes[3]
    neg_t2 = class_volumes[0] + class_volumes[1]
    conc_t2 = get_concentration(pos_t2, neg_t2)

    # Create results row
    results = {
        'file_path': save_path,
        'C_target1(copies/uL)': conc_t1,
        'C_target2(copies/uL)': conc_t2,
        'N0': len(class_volumes[0]),
        'N1': len(class_volumes[1]),
        'N2': len(class_volumes[2]),
        'N1&2': len(class_volumes[3]),
        'N_total': total_droplets,
        'avg_radius_eq(um)': avg_radius
    }

    # Update consolidated results
    if consolidated_df is None:
        consolidated_df = pd.DataFrame(columns=results.keys())

    return pd.concat([consolidated_df, pd.DataFrame([results])], ignore_index=True)


def process_video(video_path, output_path, save_dir, model, classifier, label):
    """
    Process a single video file through the analysis pipeline.

    Args:
        video_path (str): Path to input video
        output_path (str): Path for output video with detections
        save_dir (str): Directory for saving cropped droplets
        model (YOLO): Detection model
        classifier (YOLO): Classification model
        label (str): Video label identifier

    Returns:
        pd.DataFrame: Tracking information for all droplets
    """
    global track_id_counter, track_id_info

    # Initialize tracking variables
    track_id_counter = {}
    track_id_info = defaultdict(lambda: {'classes': [], 'confidences': [], 'areas': []})

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 960))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process every nth frame (where n = fps)
        if frame_count % fps != 0:
            frame_count += 1
            continue

        # Pad frame to standard size
        frame = cv2.copyMakeBorder(
            frame,
            (960 - frame.shape[0]) // 2,
            (960 - frame.shape[0]) // 2,
            (1280 - frame.shape[1]) // 2,
            (1280 - frame.shape[1]) // 2,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Enhance contrast and convert to RGB
        clahe_frame = apply_clahe(frame)
        clahe_rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)

        # Detect and track droplets
        results = model.track(clahe_rgb, persist=True)
        annotated = results[0].plot(line_width=1)
        out.write(annotated)

        # Process detected droplets
        crop_with_mask(
            frame_count, results, frame, save_dir,
            wh=CROP_SIZE, label=label, classifier=classifier
        )

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}")

    cap.release()
    out.release()

    # Compile tracking results
    tracking_data = []
    for track_id, info in track_id_info.items():
        if not info['classes']:
            continue

        # Determine dominant class
        class_counts = {}
        for class_id, conf in zip(info['classes'], info['confidences']):
            if class_id not in class_counts:
                class_counts[class_id] = {'count': 0, 'confidence': 0}
            class_counts[class_id]['count'] += 1
            class_counts[class_id]['confidence'] += conf[class_id]

        dominant_class = max(
            class_counts,
            key=lambda x: (class_counts[x]['count'], class_counts[x]['confidence'])
        )
        avg_area = np.mean(info['areas'])

        tracking_data.append({
            'track_id': track_id,
            'class': dominant_class,
            'area': avg_area
        })

    return pd.DataFrame(tracking_data)

def main():
    """Main execution function for the analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Video Tracking and Droplet Analysis Pipeline")
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory containing video files and where results will be saved')
    parser.add_argument('--video_labels', nargs='+', required=True,
                        help='List of video labels (e.g., ["0", "1", "2", "3"])')  # empty small medium large
    parser.add_argument('--detection_model', type=str, required=True,
                        help='Path to YOLO segmentation model weights')
    parser.add_argument('--classification_model', type=str, required=True,
                        help='Path to YOLO classification model weights')

    args = parser.parse_args()

    start_time = time.time()

    # Configuration
    ROOT_DIR = args.root_dir
    VIDEO_LABELS = args.video_labels  # List of video labels to process

    # Load models
    detection_model = YOLO(args.detection_model)
    classification_model = YOLO(args.classification_model)

    detection_model.eval()
    classification_model.eval()

    consolidated_results = None


    for label in VIDEO_LABELS:
        label_name = label.split('_')[0]

        video_path = os.path.join(ROOT_DIR, label_name, f"{label}.avi")
        output_path = os.path.join(ROOT_DIR, label_name, f"{label}_detected.avi")
        save_dir = os.path.join(ROOT_DIR, f"{label}_crops")
        os.makedirs(save_dir, exist_ok=True)

        # Process video
        tracking_df = process_video(
            video_path, output_path, save_dir,
            detection_model, classification_model, label_name
        )

        # Save and analyze results
        tracking_df.to_excel(os.path.join(ROOT_DIR, f"{label}_tracking.xlsx"), index=False)
        consolidated_results = quantify_targets(tracking_df, ROOT_DIR, label, consolidated_results)

    if consolidated_results is not None:
        consolidated_results.to_excel(
            os.path.join(ROOT_DIR, "consolidated_results.xlsx"),
            index=False
        )

    print(f"Processing completed in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()