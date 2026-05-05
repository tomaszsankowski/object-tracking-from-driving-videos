from pathlib import Path

import cv2
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

BDD_CSV_PATH = ROOT_DIR / "original_datasets" / "bdd100k_dataset" / "mot_labels.csv"
BDD_VIDEOS_DIR = ROOT_DIR / "original_datasets" / "bdd100k_dataset" / "bdd100k_videos_train_00" / "bdd100k" / "videos" / "train"

KITTI_LABELS_DIR = ROOT_DIR / "original_datasets" / "kitti_dataset" / "data_tracking_label_2" / "training" / "label_02"
KITTI_IMAGES_DIR = ROOT_DIR / "original_datasets" / "kitti_dataset" / "data_tracking_image_2" / "training" / "image_02"

DEFAULT_OUTPUT_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MAX_VIDEOS = None

# In BDD labels are stored for every 6th frame from the source video.
BDD_FRAME_STRIDE = 6

TARGET_COLUMNS = [
    "dataset_source",
    "video_id",
    "frame_index",
    "image_path",
    "track_id",
    "global_track_id",
    "category",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "occluded",
    "truncated",
]

KITTI_CLASS_MAP = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "cyclist": "rider",
}

BDD_CLASS_MAP = {
    "car": "car",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "rider": "rider"
}

SUPPORTED_CLASSES = {
    "car",
    "truck",
    "pedestrian",
    "rider"
}


def check_input_files():
    input_paths = {
        "BDD labels CSV": BDD_CSV_PATH,
        "BDD videos directory": BDD_VIDEOS_DIR,
        "KITTI labels directory": KITTI_LABELS_DIR,
        "KITTI images directory": KITTI_IMAGES_DIR,
    }

    missing_paths = [f"{label}: {path}" for label, path in input_paths.items() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Brak wymaganych danych wejściowych:\n- " + "\n- ".join(missing_paths))


def map_and_filter_categories(df, class_map):
    df = df.copy()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].map(class_map)
    return df.dropna(subset=["category"]).loc[lambda data: data["category"].isin(SUPPORTED_CLASSES)].copy()


def filter_valid_boxes(df, dataset_name="dataset"):
    bbox_columns = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    for column_name in bbox_columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    before = len(df)
    df = df.dropna(subset=bbox_columns)
    valid_mask = (df["bbox_x2"] > df["bbox_x1"]) & (df["bbox_y2"] > df["bbox_y1"])
    df = df[valid_mask].copy()

    removed = before - len(df)
    if removed > 0:
        print(f"[{dataset_name}] Usunięto {removed} rekordów z niepoprawnym bbox.")
    return df


def process_bdd100k(csv_path, videos_dir, output_dir, max_videos=None):
    print("\nRozpoczynam przetwarzanie BDD100K...")
    labels_df = pd.read_csv(csv_path, low_memory=False, dtype={"videoName": str, "name": str, "id": str})

    labels_df = labels_df.rename(columns={
        "videoName": "video_id",
        "frameIndex": "frame_index",
        "id": "track_id",
        "box2d.x1": "bbox_x1",
        "box2d.y1": "bbox_y1",
        "box2d.x2": "bbox_x2",
        "box2d.y2": "bbox_y2",
    })

    labels_df = map_and_filter_categories(labels_df, BDD_CLASS_MAP)
    labels_df["occluded"] = (
        (labels_df["attributes.occluded"].fillna(False).astype(str).str.lower() == "true").astype(int)
        if "attributes.occluded" in labels_df.columns else 0
    )
    labels_df["truncated"] = (
        (labels_df["attributes.truncated"].fillna(False).astype(str).str.lower() == "true").astype(int)
        if "attributes.truncated" in labels_df.columns else 0
    )
    labels_df["dataset_source"] = "BDD100K"
    labels_df["frame_index"] = pd.to_numeric(labels_df["frame_index"], errors="coerce")
    labels_df = labels_df.dropna(subset=["frame_index"]).copy()
    labels_df["frame_index"] = labels_df["frame_index"].astype(int)
    labels_df["track_id"] = labels_df["track_id"].astype(str)

    unique_videos = labels_df["video_id"].unique()
    if max_videos:
        unique_videos = unique_videos[:max_videos]
        labels_df = labels_df[labels_df["video_id"].isin(unique_videos)]

    exported_rows = []
    for video_id in unique_videos:
        mov_path = videos_dir / f"{video_id}.mov"
        if not mov_path.exists():
            print(f"BDD100K: Brak pliku wideo: {mov_path}")
            continue

        video_out_dir = output_dir / f"bdd_{video_id}"
        video_out_dir.mkdir(parents=True, exist_ok=True)

        video_labels = labels_df[labels_df["video_id"] == video_id].copy()

        cap = cv2.VideoCapture(str(mov_path))
        if not cap.isOpened():
            print(f"BDD100K: Nie można otworzyć wideo: {mov_path}")
            continue

        expected_frames = sorted(set(video_labels["frame_index"].unique()))
        saved_frames = 0

        for frame_index in expected_frames:
            raw_frame_index = frame_index * BDD_FRAME_STRIDE
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_index)
            ret, frame = cap.read()

            if not ret:
                continue

            image_name = video_labels[video_labels["frame_index"] == frame_index]["name"].iloc[0]
            image_relative_path = f"bdd_{video_id}/{image_name}"
            img_out_path = video_out_dir / image_name
            cv2.imwrite(str(img_out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            video_labels.loc[video_labels["frame_index"] == frame_index, "image_path"] = image_relative_path
            saved_frames += 1

        cap.release()
        print(f"BDD100K: Wideo {video_id} -> wyekstrahowano {saved_frames}/{len(expected_frames)} klatek")

        exported_rows.append(video_labels)

    if not exported_rows:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    merged_labels = pd.concat(exported_rows, ignore_index=True)
    merged_labels = merged_labels[merged_labels["image_path"] != ""].copy()
    merged_labels = filter_valid_boxes(merged_labels, "BDD100K")
    return merged_labels


def process_kitti(labels_dir, images_dir, output_dir, max_videos=None):
    print("\nRozpoczynam przetwarzanie KITTI...")
    kitti_columns = [
        "frame_index",
        "track_id",
        "category",
        "truncated",
        "occluded",
        "alpha",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "dim_h",
        "dim_w",
        "dim_l",
        "loc_x",
        "loc_y",
        "loc_z",
        "rot_y",
    ]

    txt_files = sorted(labels_dir.glob("*.txt"))
    if max_videos:
        txt_files = txt_files[:max_videos]

    exported_rows = []
    for file_path in txt_files:
        video_id = file_path.stem
        video_img_dir = images_dir / video_id
        if not video_img_dir.exists():
            continue

        labels_df = pd.read_csv(file_path, sep=r"\s+", names=kitti_columns, engine="python")
        labels_df = labels_df[labels_df["category"].astype(str).str.lower().str.strip() != "dontcare"].copy()
        labels_df = map_and_filter_categories(labels_df, KITTI_CLASS_MAP)
        labels_df["frame_index"] = pd.to_numeric(labels_df["frame_index"], errors="coerce")
        labels_df = labels_df.dropna(subset=["frame_index"]).copy()
        labels_df["frame_index"] = labels_df["frame_index"].astype(int)
        labels_df["track_id"] = labels_df["track_id"].astype(str)

        video_out_dir = output_dir / f"kitti_{video_id}"
        video_out_dir.mkdir(parents=True, exist_ok=True)

        labels_df["dataset_source"] = "KITTI"
        labels_df["video_id"] = video_id
        labels_df["image_path"] = ""

        saved_frames = 0
        for frame_index in labels_df["frame_index"].unique():
            src_img_name = f"{frame_index:06d}.png"
            src_img_path = video_img_dir / src_img_name

            if src_img_path.exists():
                dst_img_name = f"{frame_index:06d}.jpg"
                dst_img_path = video_out_dir / dst_img_name

                img = cv2.imread(str(src_img_path))
                if img is not None:
                    cv2.imwrite(str(dst_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    labels_df.loc[labels_df["frame_index"] == frame_index, "image_path"] = f"kitti_{video_id}/{dst_img_name}"
                    saved_frames += 1

        labels_df["occluded"] = (pd.to_numeric(labels_df["occluded"], errors="coerce").fillna(0) > 0).astype(int)
        labels_df["truncated"] = (pd.to_numeric(labels_df["truncated"], errors="coerce").fillna(0) > 0).astype(int)
        labels_df = labels_df[labels_df["image_path"] != ""].copy()

        exported_rows.append(labels_df)
        print(f"KITTI: Wideo {video_id} -> skopiowano/przekonwertowano {saved_frames} klatek")

    if not exported_rows:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    merged_labels = pd.concat(exported_rows, ignore_index=True)
    merged_labels = filter_valid_boxes(merged_labels, "KITTI")
    return merged_labels


def main():
    try:
        check_input_files()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        bdd_df = process_bdd100k(BDD_CSV_PATH, BDD_VIDEOS_DIR, OUTPUT_DIR, max_videos=MAX_VIDEOS)
        kitti_df = process_kitti(KITTI_LABELS_DIR, KITTI_IMAGES_DIR, OUTPUT_DIR, max_videos=MAX_VIDEOS)

        print("\nŁączenie datasetów...")
        merged_df = pd.concat([bdd_df, kitti_df], ignore_index=True)
        if merged_df.empty:
            raise ValueError("Po filtracji nie pozostały żadne rekordy do zapisania.")

        merged_df["global_track_id"] = (
            merged_df["dataset_source"].astype(str)
            + "_"
            + merged_df["video_id"].astype(str)
            + "_"
            + merged_df["track_id"].astype(str)
        )

        merged_df = (
            merged_df[TARGET_COLUMNS]
            .sort_values(by=["dataset_source", "video_id", "frame_index", "track_id"])
            .reset_index(drop=True)
        )

        output_csv_path = OUTPUT_DIR / "merged_dataset.csv"
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_csv_path, index=False)
        print(f"\nGotowe!")
        print(f"Foldery z klatkami: {OUTPUT_DIR}")
        print(f"CSV z etykietami: {output_csv_path}")
        print(f"Łącznie rekordów: {len(merged_df)}")

    except Exception as e:
        print(f"Błąd: {e}")
        raise


if __name__ == "__main__":
    main()