"""
Script to compare predictions from best models across splits with ground truth.
Generates 3 pairs of images (ground truth + predictions) from validation sets.
"""

import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "reports" / "model_comparison"
RANDOM_SEED = 42
CONF_THRESHOLD = 0.5

CLASS_NAMES = ["car", "pedestrian", "truck", "rider"]
COLORS = {
    "ground_truth": (0, 255, 0),      # Green
    "prediction": (0, 0, 255),         # Red
}


def load_validation_annotations(split_name: str, root_dir: Path) -> pd.DataFrame:
    """Load validation CSV for a split."""
    val_csv = root_dir / "splits" / split_name / "val.csv"
    return pd.read_csv(val_csv)


def list_image_paths(df: pd.DataFrame) -> List[str]:
    """Get unique image paths from dataframe."""
    return df["image_path"].unique().tolist()


def get_annotations_for_image(df: pd.DataFrame, image_path: str) -> List[Dict]:
    """Get all annotations for a specific image."""
    annotations = df[df["image_path"] == image_path].to_dict("records")
    return annotations


def draw_annotations(
    image: np.ndarray,
    annotations: List[Dict],
    color: Tuple[int, int, int],
    label_type: str = "gt"
) -> np.ndarray:
    """Draw bounding boxes on image."""
    image_copy = image.copy()
    
    for ann in annotations:
        x1 = int(ann["bbox_x1"])
        y1 = int(ann["bbox_y1"])
        x2 = int(ann["bbox_x2"])
        y2 = int(ann["bbox_y2"])
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        category = ann["category"]
        label = f"{category} ({label_type})"
        font_scale = 0.5
        font_thickness = 1
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, text_size[1])
        
        # Draw background for text
        cv2.rectangle(
            image_copy,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image_copy,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return image_copy


def draw_predictions(
    image: np.ndarray,
    results,
    color: Tuple[int, int, int],
    conf_threshold: float = 0.5
) -> np.ndarray:
    """Draw YOLO predictions on image."""
    image_copy = image.copy()
    
    if results.boxes is None or len(results.boxes) == 0:
        return image_copy
    
    for box in results.boxes:
        conf = box.conf.item()
        if conf < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls.item())
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        category = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        label = f"{category} {conf:.2f} (pred)"
        font_scale = 0.5
        font_thickness = 1
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, text_size[1])
        
        # Draw background for text
        cv2.rectangle(
            image_copy,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image_copy,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return image_copy


def build_comparison_images(
    original_image: np.ndarray,
    gt_annotations: List[Dict],
    yolo_results,
    conf_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create ground truth and predictions images separately."""

    gt_image = draw_annotations(original_image, gt_annotations, COLORS["ground_truth"], "GT")
    pred_image = draw_predictions(original_image, yolo_results, COLORS["prediction"], conf_threshold)

    return gt_image, pred_image


def process_split(
    split_name: str,
    model_path: Path,
    root_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    conf_threshold: float = 0.5,
    seed: int = 42
) -> Tuple[str, List[Path], str]:
    """Process a single split and return comparison image paths."""

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Processing {split_name}")
    print(f"{'='*60}")

    # Load validation data
    print(f"Loading validation data for {split_name}...")
    val_df = load_validation_annotations(split_name, root_dir)

    # Get unique images
    image_paths = list_image_paths(val_df)
    print(f"Total unique validation images: {len(image_paths)}")

    # Select random image
    selected_image = random.choice(image_paths)
    print(f"Selected image: {selected_image}")

    # Load image
    image_path = dataset_dir / selected_image
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return split_name, [], selected_image

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return split_name, [], selected_image

    print(f"Image shape: {image.shape}")

    # Get ground truth annotations
    gt_annotations = get_annotations_for_image(val_df, selected_image)
    print(f"Found {len(gt_annotations)} ground truth annotations")

    # Load model and run inference
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    print("Running inference...")
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)

    if results and len(results) > 0:
        result = results[0]
        print(f"Found {len(result.boxes) if result.boxes else 0} predictions")
    else:
        print("No predictions returned")
        result = None

    # Create output images
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    # Save as two separate images for all splits
    gt_image, pred_image = build_comparison_images(image, gt_annotations, result, conf_threshold)

    gt_path = output_dir / f"{split_name}_labels.png"
    pred_path = output_dir / f"{split_name}_pred.png"

    cv2.imwrite(str(gt_path), gt_image)
    cv2.imwrite(str(pred_path), pred_image)

    print(f"Saved labels to {gt_path}")
    print(f"Saved predictions to {pred_path}")
    output_paths = [gt_path, pred_path]

    return split_name, output_paths, selected_image


def main():
    dataset_dir = ROOT_DIR / "dataset"

    splits_config = {
        "split1": ROOT_DIR / "runs" / "yolo" / "model1_split1_overfit" / "weights" / "best.pt",
        "split2": ROOT_DIR / "runs" / "yolo" / "model2_split2-4" / "weights" / "best.pt",
        "split3": ROOT_DIR / "runs" / "yolo" / "model3_split3" / "weights" / "best.pt",
    }

    # Verify all models exist
    for split_name, model_path in splits_config.items():
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            return

    print("="*60)
    print("Model Predictions Comparison")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"Seed: {RANDOM_SEED}")

    saved_comparisons = []

    # Process each split
    for idx, (split_name, model_path) in enumerate(splits_config.items()):
        try:
            processed_split, output_paths, selected_image = process_split(
                split_name=split_name,
                model_path=model_path,
                root_dir=ROOT_DIR,
                dataset_dir=dataset_dir,
                output_dir=OUTPUT_DIR,
                conf_threshold=CONF_THRESHOLD,
                seed=RANDOM_SEED + idx  # Different seed per split for random selection
            )

            if output_paths:
                saved_comparisons.append({
                    "split": processed_split,
                    "outputs": [str(p) for p in output_paths],
                    "image": selected_image
                })
                print(f"✓ {processed_split} completed successfully")
            else:
                print(f"✗ {processed_split} failed")

        except Exception as e:
            print(f"✗ Error processing {split_name}: {e}")
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for info in saved_comparisons:
        print(f"\n{info['split']}:")
        for output in info['outputs']:
            print(f"  {output}")
        print(f"  Image: {info['image']}")

    print(f"\nComparison images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
