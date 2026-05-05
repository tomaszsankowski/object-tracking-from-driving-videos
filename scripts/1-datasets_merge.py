from pathlib import Path

import cv2
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

BDD_CSV_PATH = ROOT_DIR / "original_datasets" / "bdd100k_dataset" / "mot_labels.csv"
BDD_VIDEOS_DIR = ROOT_DIR / "original_datasets" / "bdd100k_dataset" / "bdd100k_videos_train_00" / "bdd100k" / "videos" / "train"

KITTI_LABELS_DIR = ROOT_DIR / "original_datasets" / "kitti_dataset" / "data_tracking_label_2" / "training" / "label_02"
KITTI_IMAGES_DIR = ROOT_DIR / "original_datasets" / "kitti_dataset" / "data_tracking_image_2" / "training" / "image_02"

DEFAULT_OUTPUT_DATA_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = DEFAULT_OUTPUT_DATA_DIR
MAX_VIDEOS = None

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

CLASS_MAP_KITTI = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "cyclist": "rider",
}

CLASS_MAP_BDD = {
    "car": "car",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "rider": "rider"
}

COMMON_CLASSES = {
    "car",
    "truck",
    "pedestrian",
    "rider"
}


def ensure_required_inputs_exist():
    required_paths = {
        "BDD labels CSV": BDD_CSV_PATH,
        "BDD videos directory": BDD_VIDEOS_DIR,
        "KITTI labels directory": KITTI_LABELS_DIR,
        "KITTI images directory": KITTI_IMAGES_DIR,
    }

    missing_paths = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Brak wymaganych danych wejściowych:\n- " + "\n- ".join(missing_paths))


def normalize_and_filter_category(df, category_map):
    df = df.copy()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].map(category_map)
    return df.dropna(subset=["category"]).loc[lambda x: x["category"].isin(COMMON_CLASSES)].copy()


def validate_bboxes(df, dataset_name="dataset"):
    for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
    valid_mask = (df["bbox_x2"] > df["bbox_x1"]) & (df["bbox_y2"] > df["bbox_y1"])
    df = df[valid_mask].copy()

    removed = before - len(df)
    if removed > 0:
        print(f"[{dataset_name}] Usunięto {removed} rekordów z niepoprawnym bbox.")
    return df


def normalize_and_filter_category(df, category_map):
    df = df.copy()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].map(category_map)
    return df.dropna(subset=["category"]).loc[lambda x: x["category"].isin(COMMON_CLASSES)].copy()


def process_bdd100k(csv_path, videos_dir, output_dir, max_videos=None):
    print("\nRozpoczynam przetwarzanie BDD100K...")
    df = pd.read_csv(csv_path, low_memory=False, dtype={"videoName": str, "name": str, "id": str})

    df = df.rename(columns={
        "videoName": "video_id",
        "frameIndex": "frame_index",
        "id": "track_id",
        "box2d.x1": "bbox_x1",
        "box2d.y1": "bbox_y1",
        "box2d.x2": "bbox_x2",
        "box2d.y2": "bbox_y2",
    })

    df = normalize_and_filter_category(df, CLASS_MAP_BDD)
    df["occluded"] = (df["attributes.occluded"].fillna(False).astype(str).str.lower() == "true").astype(int) if "attributes.occluded" in df.columns else 0
    df["truncated"] = (df["attributes.truncated"].fillna(False).astype(str).str.lower() == "true").astype(int) if "attributes.truncated" in df.columns else 0
    df["dataset_source"] = "BDD100K"
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")
    df = df.dropna(subset=["frame_index"]).copy()
    df["frame_index"] = df["frame_index"].astype(int)
    df["track_id"] = df["track_id"].astype(str)

    unique_videos = df["video_id"].unique()
    if max_videos:
        unique_videos = unique_videos[:max_videos]
        df = df[df["video_id"].isin(unique_videos)]

    valid_rows = []
    for vid_id in unique_videos:
        mov_path = videos_dir / f"{vid_id}.mov"
        if not mov_path.exists():
            print(f"BDD100K: Brak pliku wideo: {mov_path}")
            continue

        video_out_dir = output_dir / f"bdd_{vid_id}"
        video_out_dir.mkdir(parents=True, exist_ok=True)

        vid_labels = df[df["video_id"] == vid_id].copy()

        # Otwórz video i wyciągnij konkretne klatki (co 6-ta)
        cap = cv2.VideoCapture(str(mov_path))
        if not cap.isOpened():
            print(f"BDD100K: Nie można otworzyć wideo: {mov_path}")
            continue

        expected_frames = sorted(set(vid_labels["frame_index"].unique()))
        saved_frames = 0

        for logical_frame_index in expected_frames:
            raw_frame_index = logical_frame_index * BDD_FRAME_STRIDE
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_index)
            ret, frame = cap.read()
            
            if not ret:
                continue

            img_name = vid_labels[vid_labels["frame_index"] == logical_frame_index]["name"].iloc[0]
            img_out_path = video_out_dir / img_name
            cv2.imwrite(str(img_out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            vid_labels.loc[vid_labels["frame_index"] == logical_frame_index, "image_path"] = (
                f"bdd_{vid_id}/{img_name}"
            )
            saved_frames += 1

        cap.release()
        print(f"BDD100K: Wideo {vid_id} -> wyekstrahowano {saved_frames}/{len(expected_frames)} klatek")

        valid_rows.append(vid_labels)

    if not valid_rows:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    final_df = pd.concat(valid_rows, ignore_index=True)
    final_df = final_df[final_df["image_path"] != ""].copy()
    final_df = validate_bboxes(final_df, "BDD100K")
    return final_df


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

    valid_rows = []
    for file_path in txt_files:
        vid_id = file_path.stem
        video_img_dir = images_dir / vid_id
        if not video_img_dir.exists():
            continue

        df_temp = pd.read_csv(file_path, sep=r"\s+", names=kitti_columns, engine="python")
        df_temp = df_temp[df_temp["category"].astype(str).str.lower().str.strip() != "dontcare"].copy()
        df_temp = normalize_and_filter_category(df_temp, CLASS_MAP_KITTI)
        df_temp["frame_index"] = pd.to_numeric(df_temp["frame_index"], errors="coerce")
        df_temp = df_temp.dropna(subset=["frame_index"]).copy()
        df_temp["frame_index"] = df_temp["frame_index"].astype(int)
        df_temp["track_id"] = df_temp["track_id"].astype(str)

        video_out_dir = output_dir / f"kitti_{vid_id}"
        video_out_dir.mkdir(parents=True, exist_ok=True)

        df_temp["dataset_source"] = "KITTI"
        df_temp["video_id"] = vid_id
        df_temp["image_path"] = ""

        saved_frames = 0
        for frame_idx in df_temp["frame_index"].unique():
            src_img_name = f"{frame_idx:06d}.png"
            src_img_path = video_img_dir / src_img_name

            if src_img_path.exists():
                dst_img_name = f"{frame_idx:06d}.jpg"
                dst_img_path = video_out_dir / dst_img_name

                # Konwertuj PNG na JPG
                img = cv2.imread(str(src_img_path))
                if img is not None:
                    cv2.imwrite(str(dst_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    df_temp.loc[df_temp["frame_index"] == frame_idx, "image_path"] = f"kitti_{vid_id}/{dst_img_name}"
                    saved_frames += 1

        df_temp["occluded"] = (pd.to_numeric(df_temp["occluded"], errors="coerce").fillna(0) > 0).astype(int)
        df_temp["truncated"] = (pd.to_numeric(df_temp["truncated"], errors="coerce").fillna(0) > 0).astype(int)
        df_temp = df_temp[df_temp["image_path"] != ""].copy()

        valid_rows.append(df_temp)
        print(f"KITTI: Wideo {vid_id} -> skopiowano/przekonwertowano {saved_frames} klatek")

    if not valid_rows:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    final_df = pd.concat(valid_rows, ignore_index=True)
    final_df = validate_bboxes(final_df, "KITTI")
    return final_df


def main():
    try:
        ensure_required_inputs_exist()
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

        merged_df = merged_df[TARGET_COLUMNS].sort_values(by=["dataset_source", "video_id", "frame_index", "track_id"]).reset_index(drop=True)

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