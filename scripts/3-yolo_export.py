import json
import os
import shutil
from pathlib import Path

import cv2
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SPLITS_DIR = ROOT_DIR / "splits"
OUTPUT_DIR = ROOT_DIR / "yolo_dataset"
SPLIT_NAMES = ["split1", "split2", "split3"]
MAX_IMAGES_PER_SUBSET = None
COPY_MODE = "auto"

CLASS_NAMES = ["car", "pedestrian", "truck", "rider"]
CLASS_TO_ID = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
SUBSET_NAMES = ["train", "val", "test"]


def load_subset_annotations(split_dir, subset_name):
    subset_csv_path = split_dir / f"{subset_name}.csv"
    if not subset_csv_path.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {subset_csv_path}")

    annotations_df = pd.read_csv(subset_csv_path, low_memory=False)
    annotations_df["dataset_source"] = annotations_df["dataset_source"].astype(str)
    annotations_df["video_id"] = annotations_df["video_id"].astype(str)
    annotations_df["image_path"] = annotations_df["image_path"].astype(str)
    annotations_df["category"] = annotations_df["category"].astype(str).str.lower().str.strip()
    annotations_df = annotations_df[annotations_df["image_path"] != ""].copy()
    annotations_df = annotations_df[annotations_df["image_path"] != "nan"].copy()
    return annotations_df[annotations_df["category"].isin(CLASS_TO_ID)].copy()


def copy_or_link_image(source_path, destination_path, copy_mode):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        return

    if copy_mode in {"auto", "hardlink"}:
        try:
            os.link(source_path, destination_path)
            return
        except OSError:
            if copy_mode == "hardlink":
                raise

    shutil.copy2(source_path, destination_path)


def build_shared_asset_path(image_row, source_image_path):
    return Path(image_row.dataset_source.lower()) / str(image_row.video_id) / source_image_path.name


def build_manifest_entry(shared_relative_path):
    return f"./images/{shared_relative_path.as_posix()}"


def build_manifest_path(output_dir, split_name, subset_name):
    return output_dir / f"{split_name}_{subset_name}.txt"


def convert_bbox_to_yolo(row, width, height):
    x1 = max(0.0, min(float(width), float(row.bbox_x1)))
    y1 = max(0.0, min(float(height), float(row.bbox_y1)))
    x2 = max(0.0, min(float(width), float(row.bbox_x2)))
    y2 = max(0.0, min(float(height), float(row.bbox_y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    return x_center, y_center, box_width, box_height


def export_shared_asset(image_row, annotation_group, shared_images_dir, shared_labels_dir, copy_mode, asset_cache):
    cache_key = str(image_row.image_path)
    cached_record = asset_cache.get(cache_key)
    if cached_record is not None:
        return cached_record

    source_image_path = ROOT_DIR / "dataset" / image_row.image_path
    shared_relative_path = build_shared_asset_path(image_row, source_image_path)
    target_image_path = shared_images_dir / shared_relative_path
    target_label_path = shared_labels_dir / shared_relative_path.with_suffix(".txt")

    if target_image_path.exists() and target_label_path.exists():
        box_count = sum(1 for line in target_label_path.read_text(encoding="utf-8").splitlines() if line.strip())
        record = {
            "status": "exported",
            "dataset_source": image_row.dataset_source,
            "video_id": str(image_row.video_id),
            "frame_index": int(image_row.frame_index),
            "source_image_path": source_image_path.resolve().as_posix(),
            "shared_relative_path": shared_relative_path.as_posix(),
            "shared_image_path": target_image_path.resolve().as_posix(),
            "shared_label_path": target_label_path.resolve().as_posix(),
            "box_count": box_count,
        }
        asset_cache[cache_key] = record
        return record

    if not source_image_path.exists():
        record = {
            "status": "skipped",
            "message": f"brak obrazu {source_image_path}",
        }
        asset_cache[cache_key] = record
        return record

    image = cv2.imread(str(source_image_path))
    if image is None:
        record = {
            "status": "skipped",
            "message": f"nie udało się odczytać obrazu {source_image_path}",
        }
        asset_cache[cache_key] = record
        return record

    height, width = image.shape[:2]
    if annotation_group is None or annotation_group.empty:
        record = {
            "status": "skipped",
            "message": f"brak adnotacji dla {source_image_path}",
        }
        asset_cache[cache_key] = record
        return record

    yolo_lines = []
    valid_boxes_for_image = 0
    for annotation_row in annotation_group.itertuples(index=False):
        class_id = CLASS_TO_ID.get(annotation_row.category)
        if class_id is None:
            continue

        normalized_bbox = convert_bbox_to_yolo(annotation_row, width, height)
        if normalized_bbox is None:
            continue

        yolo_lines.append(
            f"{class_id} {normalized_bbox[0]:.6f} {normalized_bbox[1]:.6f} {normalized_bbox[2]:.6f} {normalized_bbox[3]:.6f}"
        )
        valid_boxes_for_image += 1

    if not yolo_lines:
        record = {
            "status": "skipped",
            "message": f"brak poprawnych bboxów dla {source_image_path}",
        }
        asset_cache[cache_key] = record
        return record

    copy_or_link_image(source_image_path, target_image_path, copy_mode)
    target_label_path.parent.mkdir(parents=True, exist_ok=True)
    target_label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    record = {
        "status": "exported",
        "dataset_source": image_row.dataset_source,
        "video_id": str(image_row.video_id),
        "frame_index": int(image_row.frame_index),
        "source_image_path": source_image_path.resolve().as_posix(),
        "shared_relative_path": shared_relative_path.as_posix(),
        "shared_image_path": target_image_path.resolve().as_posix(),
        "shared_label_path": target_label_path.resolve().as_posix(),
        "box_count": valid_boxes_for_image,
    }
    asset_cache[cache_key] = record
    return record


def export_subset(split_name, subset_name, subset_df, output_dir, copy_mode, asset_cache, max_images_per_subset=None):
    split_output_dir = output_dir / split_name
    manifest_path = build_manifest_path(output_dir, split_name, subset_name)
    shared_images_dir = output_dir / "images"
    shared_labels_dir = output_dir / "labels"

    unique_images = subset_df[["dataset_source", "video_id", "frame_index", "image_path"]].drop_duplicates()
    unique_images = unique_images.sort_values(by=["dataset_source", "video_id", "frame_index"])
    if max_images_per_subset is not None:
        unique_images = unique_images.head(max_images_per_subset)

    annotations_by_image = {image_path: group.copy() for image_path, group in subset_df.groupby("image_path", sort=False)}

    manifest_lines = []
    exported_index_rows = []
    exported_image_count = 0
    exported_box_count = 0
    skipped_images = 0

    for image_row in unique_images.itertuples(index=False):
        annotation_group = annotations_by_image.get(image_row.image_path)
        asset_record = export_shared_asset(
            image_row,
            annotation_group,
            shared_images_dir,
            shared_labels_dir,
            copy_mode,
            asset_cache,
        )
        if asset_record["status"] != "exported":
            skipped_images += 1
            print(f"{split_name}/{subset_name}: {asset_record['message']}")
            continue

        shared_relative_path = Path(asset_record["shared_relative_path"])
        manifest_lines.append(build_manifest_entry(shared_relative_path))

        exported_index_rows.append(
            {
                "dataset_source": image_row.dataset_source,
                "video_id": image_row.video_id,
                "frame_index": int(image_row.frame_index),
                "source_image_path": asset_record["source_image_path"],
                "shared_image_path": asset_record["shared_image_path"],
                "shared_label_path": asset_record["shared_label_path"],
                "manifest_entry": build_manifest_entry(shared_relative_path),
                "box_count": asset_record["box_count"],
            }
        )
        exported_image_count += 1
        exported_box_count += asset_record["box_count"]

    if manifest_lines:
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    if exported_index_rows:
        pd.DataFrame(exported_index_rows).to_csv(split_output_dir / f"{subset_name}_index.csv", index=False)

    return {
        "images": exported_image_count,
        "boxes": exported_box_count,
        "skipped_images": skipped_images,
    }


def save_dataset_yaml(split_output_dir, split_name):
    yaml_lines = [
        f"train: ../{split_name}_train.txt",
        f"val: ../{split_name}_val.txt",
        f"test: ../{split_name}_test.txt",
        "names:",
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {class_id}: {class_name}")

    (split_output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def save_shared_assets_index(output_dir, asset_cache):
    shared_assets = [record for record in asset_cache.values() if record["status"] == "exported"]
    if not shared_assets:
        return

    pd.DataFrame(shared_assets).sort_values(by=["dataset_source", "video_id", "frame_index"]).to_csv(
        output_dir / "shared_assets_index.csv",
        index=False,
    )


def export_split(split_name, splits_dir, output_dir, copy_mode, asset_cache, max_images_per_subset=None):
    split_dir = splits_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Brak katalogu splitu: {split_dir}")

    print(f"\nEksport splitu {split_name} do YOLO...")
    split_output_dir = output_dir / split_name
    if split_output_dir.exists():
        shutil.rmtree(split_output_dir)
    split_output_dir.mkdir(parents=True, exist_ok=True)

    export_summary = {}
    for subset_name in SUBSET_NAMES:
        subset_df = load_subset_annotations(split_dir, subset_name)
        export_summary[subset_name] = export_subset(
            split_name,
            subset_name,
            subset_df,
            output_dir,
            copy_mode,
            asset_cache,
            max_images_per_subset=max_images_per_subset,
        )

    save_dataset_yaml(split_output_dir, split_name)
    (split_output_dir / "export_summary.json").write_text(
        json.dumps(export_summary, indent=2),
        encoding="utf-8",
    )
    print(f"YOLO export {split_name}: {json.dumps(export_summary, ensure_ascii=False)}")


def main():
    if COPY_MODE not in {"auto", "copy", "hardlink"}:
        raise ValueError(f"Nieobsługiwany COPY_MODE: {COPY_MODE}")

    if not SPLITS_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu ze splitami: {SPLITS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_cache = {}
    for split_name in SPLIT_NAMES:
        export_split(
            split_name,
            SPLITS_DIR,
            OUTPUT_DIR,
            COPY_MODE,
            asset_cache,
            max_images_per_subset=MAX_IMAGES_PER_SUBSET,
        )
    save_shared_assets_index(OUTPUT_DIR, asset_cache)


if __name__ == "__main__":
    main()