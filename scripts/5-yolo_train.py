import json
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_PROJECT_DIR = ROOT_DIR / "runs" / "yolo"
DEFAULT_DATASET_ROOT = ROOT_DIR / "yolo_dataset"

BASE_TRAIN_CONFIG = {
    "model": "yolov8m.pt",
    "epochs": 20,
    "imgsz": 960,
    "batch": 12,
    "device": None,
    "project": DEFAULT_PROJECT_DIR,
    "workers": 4,
    "patience": 20,
    "fraction": 1.0,
    "freeze": 0,
    "seed": 42,
    "cache": False,
    "exist_ok": False,
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "close_mosaic": 10,
}

EXPERIMENT_PRESETS = {
    "smoke": {
        "split_name": "split1",
        "run_name": "split1_smoke",
        "epochs": 1,
        "imgsz": 640,
        "batch": 16,
        "fraction": 0.01,
        "workers": 0,
        "patience": 5,
    },
    "split1": {
        "split_name": "split1",
        "run_name": "model1_split1",
        "fraction": 0.1,
        "patience": 5,
    },
    "split1_overfit": {
        "split_name": "split1",
        "run_name": "model1_split1_overfit",
        "epochs": 80,
        "fraction": 0.02,
        "patience": 0,
        "augment": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
    },
    "split2": {
        "split_name": "split2",
        "run_name": "model2_split2",
        "fraction": 1.0,
        "patience": 5,
    },
    "split3": {
        "split_name": "split3",
        "run_name": "model3_split3",
        "fraction": 1.0,
        "patience": 5,
    },
}

CONFIG_KEYS = list(BASE_TRAIN_CONFIG.keys())
PRESET = "split1"
DATA_YAML_PATH = None
DATASET_ROOT = DEFAULT_DATASET_ROOT
RUN_NAME = None
DRY_RUN = False
# Użyteczne do szybkich lokalnych eksperymentów bez przepisywania presetów.
TRAIN_CONFIG_OVERRIDES = {}


def detect_default_device():
    try:
        import torch
    except ImportError:
        return "cpu"

    return "0" if torch.cuda.is_available() else "cpu"


def build_train_config():
    if PRESET is not None and PRESET not in EXPERIMENT_PRESETS:
        raise ValueError(f"Nieznany PRESET: {PRESET}")

    unknown_override_keys = sorted(set(TRAIN_CONFIG_OVERRIDES) - set(CONFIG_KEYS))
    if unknown_override_keys:
        raise ValueError(f"Nieznane klucze TRAIN_CONFIG_OVERRIDES: {unknown_override_keys}")

    config = BASE_TRAIN_CONFIG.copy()
    preset = EXPERIMENT_PRESETS.get(PRESET, {})
    config.update({key: value for key, value in preset.items() if key in CONFIG_KEYS})
    config.update({key: value for key, value in TRAIN_CONFIG_OVERRIDES.items() if value is not None})

    if config["device"] is None:
        config["device"] = detect_default_device()

    config["project"] = Path(config["project"]).resolve()
    return config, preset


def get_data_yaml_path(preset):
    if DATA_YAML_PATH is not None:
        return DATA_YAML_PATH.resolve()

    split_name = preset.get("split_name") if preset else None
    if split_name is None:
        raise ValueError("Ustaw DATA_YAML_PATH albo PRESET na górze pliku, żeby wyliczyć data.yaml.")

    data_path = DATASET_ROOT.resolve() / split_name / "data.yaml"
    return data_path


def get_run_name(data_path, preset):
    if RUN_NAME:
        return RUN_NAME
    if preset and preset.get("run_name"):
        return preset["run_name"]
    return data_path.resolve().parent.name


def train_model(data_path, run_name, config):
    from ultralytics import YOLO

    model = YOLO(config["model"])
    start_time = perf_counter()
    model.train(
        data=str(data_path.resolve()),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        project=str(config["project"]),
        name=run_name,
        workers=config["workers"],
        patience=config["patience"],
        fraction=config["fraction"],
        freeze=config["freeze"],
        seed=config["seed"],
        cache=config["cache"],
        exist_ok=config["exist_ok"],
        augment=config["augment"],
        mosaic=config["mosaic"],
        mixup=config["mixup"],
        copy_paste=config["copy_paste"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        fliplr=config["fliplr"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        close_mosaic=config["close_mosaic"],
        pretrained=True,
        save=True,
        val=True,
    )
    return perf_counter() - start_time


def main():
    train_config, preset = build_train_config()
    data_yaml_path = get_data_yaml_path(preset)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Brak pliku data.yaml: {data_yaml_path}")

    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Brak pakietu 'ultralytics'. Zainstaluj go w aktywnym .venv przed uruchomieniem treningu."
        ) from error

    train_config["project"].mkdir(parents=True, exist_ok=True)
    run_name = get_run_name(data_yaml_path, preset)
    run_summary = {
        "preset": PRESET,
        "data_yaml": str(data_yaml_path.resolve()),
        "run_name": run_name,
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in train_config.items()},
    }

    if DRY_RUN:
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
        return

    elapsed_seconds = train_model(data_yaml_path, run_name, train_config)

    run_dir = train_config["project"] / run_name
    best_checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"

    run_summary.update(
        {
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint.exists() else None,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint.exists() else None,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
    )
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()