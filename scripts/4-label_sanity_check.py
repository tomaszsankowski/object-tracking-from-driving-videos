import random
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

CLASS_NAMES = ["car", "pedestrian", "truck", "rider"]
COLORS = [
    (0, 165, 255),
    (50, 205, 50),
    (255, 140, 0),
    (65, 105, 225),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUBSET_NAMES = ["train", "val", "test"]
DATASET_DIR = ROOT_DIR / "yolo_dataset" / "split1"
OUTPUT_DIR = None
SAMPLES_PER_SUBSET = 5
SEQUENCES_PER_SUBSET = 0
RANDOM_SEED = 42


def resolve_manifest_entry(manifest_path, raw_entry):
    entry = raw_entry.strip()
    if not entry:
        return None

    entry_path = Path(entry)
    if entry.startswith("./"):
        return (manifest_path.parent / entry[2:]).resolve()
    if entry_path.is_absolute():
        return entry_path
    return (manifest_path.parent / entry).resolve()


def list_subset_images(manifest_path):
    if not manifest_path.exists():
        return []

    image_paths = []
    for raw_entry in manifest_path.read_text(encoding="utf-8").splitlines():
        resolved_path = resolve_manifest_entry(manifest_path, raw_entry)
        if resolved_path is None:
            continue
        if resolved_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(resolved_path)
    return sorted(image_paths)


def get_manifest_path(dataset_dir, subset_name):
    # Eksport w zależności od wersji zapisywał manifest albo w katalogu splitu, albo obok niego.
    split_local_manifest = dataset_dir / f"{subset_name}.txt"
    if split_local_manifest.exists():
        return split_local_manifest

    shared_manifest = dataset_dir.parent / f"{dataset_dir.name}_{subset_name}.txt"
    if shared_manifest.exists():
        return shared_manifest

    return split_local_manifest


def group_images_by_sequence(dataset_dir, image_paths):
    dataset_root = dataset_dir.parent
    images_root = (dataset_root / "images").resolve()
    grouped_sequences = {}

    for image_path in image_paths:
        try:
            relative_image_path = image_path.resolve().relative_to(images_root)
        except ValueError:
            continue

        if len(relative_image_path.parts) < 3:
            continue

        sequence_key = Path(*relative_image_path.parts[:2]).as_posix()
        grouped_sequences.setdefault(sequence_key, []).append(relative_image_path)

    return {
        sequence_key: sorted(relative_image_paths, key=lambda path: path.name)
        for sequence_key, relative_image_paths in grouped_sequences.items()
    }


def get_label_path(dataset_root, image_path):
    images_root = (dataset_root / "images").resolve()
    try:
        relative_image_path = image_path.resolve().relative_to(images_root)
        return dataset_root / "labels" / relative_image_path.with_suffix(".txt")
    except ValueError:
        normalized_path = image_path.resolve().as_posix()
        if "/images/" not in normalized_path:
            return None
        return Path(normalized_path.replace("/images/", "/labels/", 1)).with_suffix(".txt")


def denormalize_box(values, width, height):
    x_center, y_center, box_width, box_height = values
    x1 = int((x_center - box_width / 2.0) * width)
    y1 = int((y_center - box_height / 2.0) * height)
    x2 = int((x_center + box_width / 2.0) * width)
    y2 = int((y_center + box_height / 2.0) * height)
    return x1, y1, x2, y2


def draw_labels_from_file(image, label_path):
    height, width = image.shape[:2]
    if not label_path.exists():
        return image

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        normalized_values = [float(value) for value in parts[1:]]
        x1, y1, x2, y2 = denormalize_box(normalized_values, width, height)
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            CLASS_NAMES[class_id],
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return image


def render_image(dataset_root, image_path, output_path):
    label_path = get_label_path(dataset_root, image_path)
    image = cv2.imread(str(image_path))
    if image is None or label_path is None:
        return False

    rendered_image = draw_labels_from_file(image, label_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered_image)
    return True


def render_random_samples(dataset_dir, output_dir, subset_name, samples_per_subset, rng):
    manifest_path = get_manifest_path(dataset_dir, subset_name)
    dataset_root = dataset_dir.parent
    images_root = dataset_root / "images"
    if not manifest_path.exists():
        print(f"Pomijam {subset_name}: brak manifestu {manifest_path.name}.")
        return 0

    image_paths = list_subset_images(manifest_path)
    if not image_paths:
        print(f"Pomijam {subset_name}: brak obrazów.")
        return 0

    sample_size = min(samples_per_subset, len(image_paths))
    selected_images = rng.sample(image_paths, sample_size)
    rendered_count = 0

    for image_path in selected_images:
        try:
            relative_image_path = image_path.resolve().relative_to(images_root.resolve())
        except ValueError:
            relative_image_path = Path(image_path.name)

        output_path = (output_dir / subset_name / relative_image_path).with_suffix(".jpg")
        if render_image(dataset_root, image_path, output_path):
            rendered_count += 1

    print(f"Sanity check {subset_name}: zapisano {rendered_count} podglądów.")
    return rendered_count


def render_sequence_samples(dataset_dir, output_dir, subset_name, sequences_per_subset, rng):
    manifest_path = get_manifest_path(dataset_dir, subset_name)
    dataset_root = dataset_dir.parent
    images_root = (dataset_root / "images").resolve()
    if not manifest_path.exists():
        print(f"Pomijam {subset_name}: brak manifestu {manifest_path.name}.")
        return 0

    image_paths = list_subset_images(manifest_path)
    if not image_paths:
        print(f"Pomijam {subset_name}: brak obrazów.")
        return 0

    grouped_sequences = group_images_by_sequence(dataset_dir, image_paths)
    if not grouped_sequences:
        print(f"Pomijam {subset_name}: brak pełnych sekwencji do renderu.")
        return 0

    sequence_keys = sorted(grouped_sequences)
    selected_sequence_count = min(sequences_per_subset, len(sequence_keys))
    selected_sequence_keys = rng.sample(sequence_keys, selected_sequence_count)

    rendered_count = 0
    for sequence_key in selected_sequence_keys:
        relative_image_paths = grouped_sequences[sequence_key]
        for relative_image_path in relative_image_paths:
            image_path = images_root / relative_image_path
            output_path = (output_dir / subset_name / relative_image_path).with_suffix(".jpg")
            if render_image(dataset_root, image_path, output_path):
                rendered_count += 1

        print(
            f"Sanity check {subset_name}: zapisano sekwencje {sequence_key} "
            f"({len(relative_image_paths)} klatek)."
        )

    print(f"Sanity check {subset_name}: zapisano {rendered_count} klatek sekwencyjnych.")
    return rendered_count


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu datasetu: {DATASET_DIR}")

    output_dir = OUTPUT_DIR or (DATASET_DIR / "sanity_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    total_rendered = 0
    for subset_name in SUBSET_NAMES:
        if SAMPLES_PER_SUBSET > 0:
            total_rendered += render_random_samples(DATASET_DIR, output_dir / "samples", subset_name, SAMPLES_PER_SUBSET, rng)
        if SEQUENCES_PER_SUBSET > 0:
            total_rendered += render_sequence_samples(
                DATASET_DIR,
                output_dir / "sequences",
                subset_name,
                SEQUENCES_PER_SUBSET,
                rng,
            )

    print(f"Gotowe. Łącznie zapisano {total_rendered} obrazów sanity-check.")


if __name__ == "__main__":
    main()