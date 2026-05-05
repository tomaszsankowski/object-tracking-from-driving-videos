# object-tracking-from-driving-videos
Detecting and tracking objects (cars, pedestrians, trucks, etc.) in dashcam driving footage - first as static detection on individual frames, then as continuous multi-object tracking across video sequences

## Configuration
Pipeline scripts in `scripts/` no longer use CLI flags.

If you want to change inputs, outputs, or run parameters, edit the constants near the top of the matching script and then run it directly, for example:

```powershell
python scripts/1-datasets_merge.py
python scripts/2-data_split.py
python scripts/3-yolo_export.py
python scripts/4-label_sanity_check.py
python scripts/5-yolo_train.py
python scripts/6-collect_task3_report.py
```

Each script keeps its own fixed settings in code, so there is no separate pipeline config file to maintain.

