"""
Script to compare predictions from best models across splits with ground truth.
Generates 3 pairs of images (ground truth + predictions) from validation sets.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

CLASS_NAMES = ["car", "pedestrian", "truck", "rider"]
COLORS = {
    "ground_truth": (0, 255, 0),      # Green
    "prediction": (0, 0, 255),         # Red
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare best models from splits with ground truth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model_comparison"),
        help="Directory to save comparison images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions"
    )
    return parser.parse_args()


def load_val_data(split_name: str, root_dir: Path) -> pd.DataFrame:
    """Load validation CSV for a split."""
    val_csv = root_dir / "splits" / split_name / "val.csv"
    return pd.read_csv(val_csv)


def get_unique_images(df: pd.DataFrame) -> List[str]:
    """Get unique image paths from dataframe."""
    return df["image_path"].unique().tolist()


def get_image_annotations(df: pd.DataFrame, image_path: str) -> List[Dict]:
    """Get all annotations for a specific image."""
    annotations = df[df["image_path"] == image_path].to_dict("records")
    return annotations


def draw_bboxes(
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


def draw_yolo_predictions(
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


def create_comparison_pair(
    original_image: np.ndarray,
    gt_annotations: List[Dict],
    yolo_results,
    conf_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create ground truth and predictions images separately."""
    
    # Draw ground truth
    gt_image = draw_bboxes(original_image, gt_annotations, COLORS["ground_truth"], "GT")
    
    # Draw predictions
    pred_image = draw_yolo_predictions(original_image, yolo_results, COLORS["prediction"], conf_threshold)
    
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
    val_df = load_val_data(split_name, root_dir)
    
    # Get unique images
    unique_images = get_unique_images(val_df)
    print(f"Total unique validation images: {len(unique_images)}")
    
    # Select random image
    selected_image = random.choice(unique_images)
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
    gt_annotations = get_image_annotations(val_df, selected_image)
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
    gt_image, pred_image = create_comparison_pair(image, gt_annotations, result, conf_threshold)
    
    gt_path = output_dir / f"{split_name}_labels.png"
    pred_path = output_dir / f"{split_name}_pred.png"
    
    cv2.imwrite(str(gt_path), gt_image)
    cv2.imwrite(str(pred_path), pred_image)
    
    print(f"Saved labels to {gt_path}")
    print(f"Saved predictions to {pred_path}")
    output_paths = [gt_path, pred_path]
    
    return split_name, output_paths, selected_image


def main():
    args = parse_args()
    
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    dataset_dir = root_dir / "dataset"
    
    splits_config = {
        "split1": root_dir / "runs" / "yolo" / "model1_split1_overfit" / "weights" / "best.pt",
        "split2": root_dir / "runs" / "yolo" / "model2_split2-4" / "weights" / "best.pt",
        "split3": root_dir / "runs" / "yolo" / "model3_split3" / "weights" / "best.pt",
    }
    
    # Verify all models exist
    for split_name, model_path in splits_config.items():
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            return
    
    print("="*60)
    print("Model Predictions Comparison")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Seed: {args.seed}")
    
    results_info = []
    
    # Process each split
    for idx, (split_name, model_path) in enumerate(splits_config.items()):
        try:
            split_result, output_paths, selected_image = process_split(
                split_name=split_name,
                model_path=model_path,
                root_dir=root_dir,
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                conf_threshold=args.conf,
                seed=args.seed + idx  # Different seed per split for random selection
            )
            
            if output_paths:
                results_info.append({
                    "split": split_result,
                    "outputs": [str(p) for p in output_paths],
                    "image": selected_image
                })
                print(f"✓ {split_result} completed successfully")
            else:
                print(f"✗ {split_result} failed")
        
        except Exception as e:
            print(f"✗ Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for info in results_info:
        print(f"\n{info['split']}:")
        for output in info['outputs']:
            print(f"  {output}")
        print(f"  Image: {info['image']}")
    
    print(f"\nComparison images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
