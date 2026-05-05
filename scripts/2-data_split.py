import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_CSV_PATH = ROOT_DIR / "dataset" / "merged_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "splits"
CLASS_NAMES = ["car", "pedestrian", "truck", "rider"]
RANDOM_SEED = 42


def load_annotations(csv_path):
    annotations_df = pd.read_csv(csv_path, low_memory=False)
    annotations_df["dataset_source"] = annotations_df["dataset_source"].astype(str)
    annotations_df["category"] = annotations_df["category"].astype(str).str.lower().str.strip()
    annotations_df = annotations_df[annotations_df["category"].isin(CLASS_NAMES)].copy()
    annotations_df["video_id"] = annotations_df["video_id"].astype(str)
    annotations_df["sequence_id"] = annotations_df["dataset_source"] + "::" + annotations_df["video_id"]
    return annotations_df


def split_sequences_randomly(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    sequence_ids = df['sequence_id'].unique()

    train_ids, temp_ids = train_test_split(sequence_ids, train_size=train_size, random_state=random_state)

    rel_val_size = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(temp_ids, train_size=rel_val_size, random_state=random_state)

    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


def split_sequences_stratified(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    class_counts = df['category'].value_counts()
    class_order = class_counts.index.tolist()[::-1]

    sequence_tags = {}
    grouped = df.groupby('sequence_id')['category'].unique()

    for sequence_id, categories in grouped.items():
        for class_name in class_order:
            if class_name in categories:
                sequence_tags[sequence_id] = class_name
                break

    sequence_df = pd.DataFrame(list(sequence_tags.items()), columns=['sequence_id', 'tag'])

    tag_counts = sequence_df['tag'].value_counts()
    sequence_df['tag'] = sequence_df['tag'].apply(lambda tag: tag if tag_counts[tag] > 2 else 'other')

    split_tags = sequence_df['tag']
    stratify_labels = split_tags if split_tags.value_counts().min() >= 2 else None
    if stratify_labels is None:
        print("SPLIT 2: Ostrzeżenie - za mało sekwencji dla pełnej stratyfikacji. Używam podziału bez stratify.")

    train_ids, temp_ids, _, temp_tags = train_test_split(
        sequence_df['sequence_id'], sequence_df['tag'],
        train_size=train_size, stratify=stratify_labels, random_state=random_state
    )

    rel_val_size = val_size / (val_size + test_size)
    temp_stratify_labels = temp_tags if temp_tags.value_counts().min() >= 2 else None
    if temp_stratify_labels is None:
        print("SPLIT 2: Ostrzeżenie - część VAL/TEST jest zbyt mała dla pełnej stratyfikacji. Używam podziału bez stratify.")
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=rel_val_size, stratify=temp_stratify_labels, random_state=random_state
    )

    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


def print_split_summary(split_name, train_df, val_df, test_df):
    print(f"\n{'=' * 70}\nRAPORT: {split_name}\n{'=' * 70}")

    train_counts = train_df['category'].value_counts().reindex(CLASS_NAMES).fillna(0)
    val_counts = val_df['category'].value_counts().reindex(CLASS_NAMES).fillna(0)
    test_counts = test_df['category'].value_counts().reindex(CLASS_NAMES).fillna(0)

    stats_df = pd.DataFrame({'TRAIN (Ilość)': train_counts, 'VAL (Ilość)': val_counts, 'TEST (Ilość)': test_counts})

    stats_df['TRAIN (%)'] = (stats_df['TRAIN (Ilość)'] / stats_df['TRAIN (Ilość)'].sum() * 100).round(1)
    stats_df['VAL (%)'] = (stats_df['VAL (Ilość)'] / stats_df['VAL (Ilość)'].sum() * 100).round(1)
    stats_df['TEST (%)'] = (stats_df['TEST (Ilość)'] / stats_df['TEST (Ilość)'].sum() * 100).round(1)

    stats_df = stats_df[['TRAIN (Ilość)', 'TRAIN (%)', 'VAL (Ilość)', 'VAL (%)', 'TEST (Ilość)', 'TEST (%)']]
    print(stats_df)

    train_sequences = set(train_df['sequence_id'].unique())
    val_sequences = set(val_df['sequence_id'].unique())
    test_sequences = set(test_df['sequence_id'].unique())

    leak_val = train_sequences.intersection(val_sequences)
    leak_test = train_sequences.intersection(test_sequences)

    print(f"\n[Weryfikacja Wycieków - {split_name}]")
    if "VAL is subset of TRAIN" in split_name:
        print(f"Filmy w TRAIN i VAL pokrywają się celowo. Wspólne wideo: {len(leak_val)}")
        print(f"Wyciek do TEST: {len(leak_test)}")
    else:
        print(f"Wyciek TRAIN <-> VAL:  {len(leak_val)}")
        print(f"Wyciek TRAIN <-> TEST: {len(leak_test)}")

    print(f"Sekwencje TRAIN/VAL/TEST: {len(train_sequences)}/{len(val_sequences)}/{len(test_sequences)}")
    return stats_df


def save_class_distribution_plot(split_name, train_df, val_df, test_df, output_path):
    stats_df = pd.DataFrame({
        "TRAIN": train_df["category"].value_counts(normalize=True) * 100,
        "VAL": val_df["category"].value_counts(normalize=True) * 100,
        "TEST": test_df["category"].value_counts(normalize=True) * 100,
    }).reindex(CLASS_NAMES).fillna(0)

    ax = stats_df.plot(kind="bar", figsize=(10, 6), alpha=0.85)
    ax.set_title(f"Rozkład klas w zbiorach (%) - {split_name}", fontsize=14)
    ax.set_ylabel("Udział klasy [%]", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title="Zbiór")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_split_outputs(split_name, train_df, val_df, test_df, output_dir, input_csv_path, random_state, val_subset_of_train=False):
    split_dir = output_dir / split_name.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    subsets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for subset_name, subset_df in subsets.items():
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
            for subset_name, subset_df in subsets.items()
        },
    }

    with (split_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    save_class_distribution_plot(split_name.upper(), train_df, val_df, test_df, split_dir / "class_distribution.png")


def main():
    if not INPUT_CSV_PATH.exists():
        print(f"Brak pliku: {INPUT_CSV_PATH}")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotations_df = load_annotations(INPUT_CSV_PATH)

    # SPLIT 1 (Zwykły podział losowy, bez stratyfikacji)
    random_train_ids, random_val_ids, random_test_ids = split_sequences_randomly(annotations_df, random_state=RANDOM_SEED)

    split1_train = annotations_df[annotations_df["sequence_id"].isin(random_train_ids)].copy()
    split1_val = annotations_df[annotations_df["sequence_id"].isin(random_val_ids)].copy()
    split1_test = annotations_df[annotations_df["sequence_id"].isin(random_test_ids)].copy()

    print_split_summary("SPLIT 1 (Original Data - No Stratification)", split1_train, split1_val, split1_test)
    save_split_outputs("split1", split1_train, split1_val, split1_test, OUTPUT_DIR, INPUT_CSV_PATH, RANDOM_SEED)

    # SPLIT 2 (Stratyfikacja klas Rarest-Class-First)
    strat_train_ids, strat_val_ids, strat_test_ids = split_sequences_stratified(annotations_df, random_state=RANDOM_SEED)

    split2_train = annotations_df[annotations_df["sequence_id"].isin(strat_train_ids)].copy()
    split2_val = annotations_df[annotations_df["sequence_id"].isin(strat_val_ids)].copy()
    split2_test = annotations_df[annotations_df["sequence_id"].isin(strat_test_ids)].copy()

    print_split_summary("SPLIT 2 (Properly Normalized/Stratified & Ready for Augmentation)", split2_train, split2_val,
                      split2_test)
    save_split_outputs("split2", split2_train, split2_val, split2_test, OUTPUT_DIR, INPUT_CSV_PATH, RANDOM_SEED)

    # SPLIT 3 (VAL is subset of TRAIN)
    split3_train = split2_train.copy()
    split3_test = split2_test.copy()

    split3_train_ids = split3_train['sequence_id'].unique()
    _, split3_val_ids = train_test_split(split3_train_ids, test_size=0.2, random_state=RANDOM_SEED)
    split3_val = split3_train[split3_train['sequence_id'].isin(split3_val_ids)].copy()

    print_split_summary("SPLIT 3 (VAL is subset of TRAIN)", split3_train, split3_val, split3_test)
    save_split_outputs(
        "split3",
        split3_train,
        split3_val,
        split3_test,
        OUTPUT_DIR,
        INPUT_CSV_PATH,
        RANDOM_SEED,
        val_subset_of_train=True,
    )


if __name__ == "__main__":
    main()