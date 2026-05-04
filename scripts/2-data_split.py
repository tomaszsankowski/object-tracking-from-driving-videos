import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(BASE_DIR).resolve().parent
MERGED_DATASET_PATH = ROOT_DIR / "dataset/merged_dataset.csv"
DEFAULT_SPLITS_DIR = ROOT_DIR / "splits"
COMMON_CLASSES = ["car", "pedestrian", "truck", "rider"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create reusable SPLIT1/SPLIT2/SPLIT3 artifacts for Task 3.")
    parser.add_argument("--input-csv", type=Path, default=MERGED_DATASET_PATH, help="Path to merged_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Directory for persisted split artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for all splits")
    return parser.parse_args()


def load_dataset(path):
    df = pd.read_csv(path, low_memory=False)
    df["dataset_source"] = df["dataset_source"].astype(str)
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df = df[df["category"].isin(COMMON_CLASSES)].copy()
    df["video_id"] = df["video_id"].astype(str)
    df["sequence_id"] = df["dataset_source"] + "::" + df["video_id"]
    return df


def get_random_video_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    unique_vids = df['sequence_id'].unique()

    train_vids, temp_vids = train_test_split(unique_vids, train_size=train_size, random_state=random_state)

    rel_val_size = val_size / (val_size + test_size)
    val_vids, test_vids = train_test_split(temp_vids, train_size=rel_val_size, random_state=random_state)

    return train_vids.tolist(), val_vids.tolist(), test_vids.tolist()


def get_stratified_video_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    class_counts = df['category'].value_counts()
    class_order = class_counts.index.tolist()[::-1]

    video_tags = {}
    grouped = df.groupby('sequence_id')['category'].unique()

    for vid, categories in grouped.items():
        for c in class_order:
            if c in categories:
                video_tags[vid] = c
                break

    video_df = pd.DataFrame(list(video_tags.items()), columns=['video_id', 'tag'])

    val_counts = video_df['tag'].value_counts()
    video_df['tag'] = video_df['tag'].apply(lambda x: x if val_counts[x] > 2 else 'other')

    split_tags = video_df['tag']
    stratify_labels = split_tags if split_tags.value_counts().min() >= 2 else None
    if stratify_labels is None:
        print("SPLIT 2: Ostrzeżenie - za mało sekwencji dla pełnej stratyfikacji. Używam podziału bez stratify.")

    train_vids, temp_vids, _, temp_tags = train_test_split(
        video_df['video_id'], video_df['tag'],
        train_size=train_size, stratify=stratify_labels, random_state=random_state
    )

    rel_val_size = val_size / (val_size + test_size)
    temp_stratify_labels = temp_tags if temp_tags.value_counts().min() >= 2 else None
    if temp_stratify_labels is None:
        print("SPLIT 2: Ostrzeżenie - część VAL/TEST jest zbyt mała dla pełnej stratyfikacji. Używam podziału bez stratify.")
    val_vids, test_vids = train_test_split(
        temp_vids, train_size=rel_val_size, stratify=temp_stratify_labels, random_state=random_state
    )

    return train_vids.tolist(), val_vids.tolist(), test_vids.tolist()


def print_split_stats(name, train_df, val_df, test_df):
    print(f"\n{'=' * 70}\nRAPORT: {name}\n{'=' * 70}")

    train_c = train_df['category'].value_counts().reindex(COMMON_CLASSES).fillna(0)
    val_c = val_df['category'].value_counts().reindex(COMMON_CLASSES).fillna(0)
    test_c = test_df['category'].value_counts().reindex(COMMON_CLASSES).fillna(0)

    stats = pd.DataFrame({'TRAIN (Ilość)': train_c, 'VAL (Ilość)': val_c, 'TEST (Ilość)': test_c})

    stats['TRAIN (%)'] = (stats['TRAIN (Ilość)'] / stats['TRAIN (Ilość)'].sum() * 100).round(1)
    stats['VAL (%)'] = (stats['VAL (Ilość)'] / stats['VAL (Ilość)'].sum() * 100).round(1)
    stats['TEST (%)'] = (stats['TEST (Ilość)'] / stats['TEST (Ilość)'].sum() * 100).round(1)

    stats = stats[['TRAIN (Ilość)', 'TRAIN (%)', 'VAL (Ilość)', 'VAL (%)', 'TEST (Ilość)', 'TEST (%)']]
    print(stats)

    train_vids = set(train_df['sequence_id'].unique())
    val_vids = set(val_df['sequence_id'].unique())
    test_vids = set(test_df['sequence_id'].unique())

    leak_val = train_vids.intersection(val_vids)
    leak_test = train_vids.intersection(test_vids)

    print(f"\n[Weryfikacja Wycieków - {name}]")
    if "VAL is subset of TRAIN" in name:
        print(f"Filmy w TRAIN i VAL pokrywają się celowo. Wspólne wideo: {len(leak_val)}")
        print(f"Wyciek do TEST: {len(leak_test)}")
    else:
        print(f"Wyciek TRAIN <-> VAL:  {len(leak_val)}")
        print(f"Wyciek TRAIN <-> TEST: {len(leak_test)}")

    print(f"Sekwencje TRAIN/VAL/TEST: {len(train_vids)}/{len(val_vids)}/{len(test_vids)}")
    return stats


def plot_split_proportions(name, train_df, val_df, test_df, filename):
    stats = pd.DataFrame({
        "TRAIN": train_df["category"].value_counts(normalize=True) * 100,
        "VAL": val_df["category"].value_counts(normalize=True) * 100,
        "TEST": test_df["category"].value_counts(normalize=True) * 100,
    }).reindex(COMMON_CLASSES).fillna(0)

    ax = stats.plot(kind="bar", figsize=(10, 6), alpha=0.85)
    ax.set_title(f"Rozkład klas w zbiorach (%) - {name}", fontsize=14)
    ax.set_ylabel("Udział klasy [%]", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title="Zbiór")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_split_artifacts(split_name, train_df, val_df, test_df, output_dir, input_csv_path, random_state, val_subset_of_train=False):
    split_dir = output_dir / split_name.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    subset_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for subset_name, subset_df in subset_frames.items():
        subset_df.sort_values(by=["dataset_source", "video_id", "frame_index", "track_id"]).to_csv(
            split_dir / f"{subset_name}.csv", index=False
        )
        subset_df[["dataset_source", "video_id", "sequence_id"]].drop_duplicates().sort_values(
            by=["dataset_source", "video_id"]
        ).to_csv(split_dir / f"{subset_name}_sequences.csv", index=False)

    stats = pd.DataFrame({
        "TRAIN": train_df["category"].value_counts().reindex(COMMON_CLASSES).fillna(0).astype(int),
        "VAL": val_df["category"].value_counts().reindex(COMMON_CLASSES).fillna(0).astype(int),
        "TEST": test_df["category"].value_counts().reindex(COMMON_CLASSES).fillna(0).astype(int),
    })
    stats.to_csv(split_dir / "class_distribution.csv")

    manifest = {
        "split_name": split_name,
        "input_csv": str(input_csv_path.resolve()),
        "random_state": random_state,
        "val_subset_of_train": val_subset_of_train,
        "subsets": {
            subset_name: {
                "rows": int(len(subset_df)),
                "sequences": sorted(subset_df["sequence_id"].drop_duplicates().tolist()),
            }
            for subset_name, subset_df in subset_frames.items()
        },
    }

    with (split_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    plot_split_proportions(split_name.upper(), train_df, val_df, test_df, split_dir / "class_distribution.png")


def main():
    args = parse_args()

    if not args.input_csv.exists():
        print(f"Brak pliku: {args.input_csv}")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_base = load_dataset(args.input_csv)

    # SPLIT 1 (Zwykły podział losowy, bez stratyfikacji)
    rand_tr_ids, rand_val_ids, rand_ts_ids = get_random_video_splits(df_base, random_state=args.random_state)

    split1_train = df_base[df_base["sequence_id"].isin(rand_tr_ids)].copy()
    split1_val = df_base[df_base["sequence_id"].isin(rand_val_ids)].copy()
    split1_test = df_base[df_base["sequence_id"].isin(rand_ts_ids)].copy()

    print_split_stats("SPLIT 1 (Original Data - No Stratification)", split1_train, split1_val, split1_test)
    save_split_artifacts("split1", split1_train, split1_val, split1_test, args.output_dir, args.input_csv, args.random_state)

    # SPLIT 2 (Stratyfikacja klas Rarest-Class-First)
    strat_tr_ids, strat_val_ids, strat_ts_ids = get_stratified_video_splits(df_base, random_state=args.random_state)

    split2_train = df_base[df_base["sequence_id"].isin(strat_tr_ids)].copy()
    split2_val = df_base[df_base["sequence_id"].isin(strat_val_ids)].copy()
    split2_test = df_base[df_base["sequence_id"].isin(strat_ts_ids)].copy()

    print_split_stats("SPLIT 2 (Properly Normalized/Stratified & Ready for Augmentation)", split2_train, split2_val,
                      split2_test)
    save_split_artifacts("split2", split2_train, split2_val, split2_test, args.output_dir, args.input_csv, args.random_state)

    # SPLIT 3 (VAL is subset of TRAIN)
    split3_train = split2_train.copy()
    split3_test = split2_test.copy()

    split3_tr_ids = split3_train['sequence_id'].unique()
    _, split3_val_ids = train_test_split(split3_tr_ids, test_size=0.2, random_state=args.random_state)
    split3_val = split3_train[split3_train['sequence_id'].isin(split3_val_ids)].copy()

    print_split_stats("SPLIT 3 (VAL is subset of TRAIN)", split3_train, split3_val, split3_test)
    save_split_artifacts(
        "split3",
        split3_train,
        split3_val,
        split3_test,
        args.output_dir,
        args.input_csv,
        args.random_state,
        val_subset_of_train=True,
    )


if __name__ == "__main__":
    main()