import argparse
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


def detect_default_device():
    try:
        import torch
    except ImportError:
        return "cpu"

    return "0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO detector on an exported split.")
    parser.add_argument("--data", type=Path, default=None, help="Path to a YOLO data.yaml file")
    parser.add_argument(
        "--preset",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default=None,
        help="Named Task 3 training preset (split1, split1_overfit, split2, split3, smoke)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing exported YOLO split folders",
    )
    parser.add_argument("--model", default=None, help="Ultralytics model checkpoint name or path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--device", default=None, help="Training device, for example cpu, 0, 0,1")
    parser.add_argument("--project", type=Path, default=None, help="Output project directory")
    parser.add_argument("--name", default=None, help="Run name inside the project directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of the training set to use")
    parser.add_argument("--freeze", type=int, default=None, help="Number of layers to freeze")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None, help="Enable Ultralytics image caching")
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=None, help="Allow reusing an existing run directory")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable data augmentation")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation strength")
    parser.add_argument("--mixup", type=float, default=None, help="Mixup augmentation strength")
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=None, help="Copy-paste augmentation strength")
    parser.add_argument("--degrees", type=float, default=None, help="Random rotation range")
    parser.add_argument("--translate", type=float, default=None, help="Random translation factor")
    parser.add_argument("--scale", type=float, default=None, help="Random scale factor")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability")
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=None, help="HSV hue augmentation strength")
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=None, help="HSV saturation augmentation strength")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=None, help="HSV value augmentation strength")
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=None, help="Epoch after which mosaic is disabled")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configuration and print it without running training")
    return parser.parse_args()


def resolve_config(args):
    config = BASE_TRAIN_CONFIG.copy()
    preset = EXPERIMENT_PRESETS.get(args.preset, {})
    config.update({key: value for key, value in preset.items() if key in CONFIG_KEYS})

    for key in CONFIG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if config["device"] is None:
        config["device"] = detect_default_device()

    config["project"] = Path(config["project"]).resolve()
    return config, preset


def resolve_data_path(args, preset):
    if args.data is not None:
        return args.data.resolve()

    split_name = preset.get("split_name") if preset else None
    if split_name is None:
        raise ValueError("Podaj --data albo wybierz --preset, z którego można wyliczyć data.yaml.")

    data_path = args.dataset_root.resolve() / split_name / "data.yaml"
    return data_path


def resolve_run_name(args, data_path, preset):
    if args.name:
        return args.name
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
    args = parse_args()
    config, preset = resolve_config(args)
    data_path = resolve_data_path(args, preset)
    if not data_path.exists():
        raise FileNotFoundError(f"Brak pliku data.yaml: {data_path}")

    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Brak pakietu 'ultralytics'. Zainstaluj go w aktywnym .venv przed uruchomieniem treningu."
        ) from error

    config["project"].mkdir(parents=True, exist_ok=True)
    run_name = resolve_run_name(args, data_path, preset)
    summary = {
        "preset": args.preset,
        "data_yaml": str(data_path.resolve()),
        "run_name": run_name,
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in config.items()},
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    elapsed_seconds = train_model(data_path, run_name, config)

    run_dir = config["project"] / run_name
    best_checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"

    summary.update(
        {
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint.exists() else None,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint.exists() else None,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
    )
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()