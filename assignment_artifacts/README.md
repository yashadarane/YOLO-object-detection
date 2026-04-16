# Multi-Object Detection Assignment

This folder contains a finished assignment setup built from the original `Senior-Design-VIAD-4` dataset using only **500 images** and **5 important classes**:

- `person`
- `car`
- `bicycle`
- `bus`
- `traffic_light`

## Why these 5 classes

These classes were selected because they are both common in the dataset and meaningful for a practical street-scene detection task:

- `person`: highest-priority safety class
- `car`: most frequent vehicle class
- `bicycle`: important small moving road object
- `bus`: large public transport object
- `traffic_light`: useful traffic-scene control signal

The original dataset has noisy labels such as `Person/person` and `green_light/red_light/yellow_light/traffic light`. They were normalized into cleaner project classes.

## Dataset created

The subset is stored in [viad_5class_500](./viad_5class_500) in YOLO format.

- Train: 400 images
- Valid: 50 images
- Test: 50 images

Main files:

- `viad_5class_500/data.yaml`
- `viad_5class_500/selection_summary.json`
- `colab_train_yolov8.py`
- `inference_with_custom_feature.py`
- `fastapi_app.py`
- `requirements-fastapi.txt`
- `FASTAPI_USAGE.md`
- `report.md`

## Google Colab workflow

Because your laptop only has CPU, use Google Colab GPU:

1. Open a new Colab notebook.
2. Set runtime to `GPU` from `Runtime > Change runtime type`.
3. Upload this project folder to Google Drive or upload the required files directly.
4. In Colab, run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Change into the folder that contains this assignment:

```python
%cd /content/drive/MyDrive/Allianz/DL
```

6. Install YOLOv8:

```python
!pip install -q ultralytics
```

7. Run training:

```python
!python assignment_artifacts/colab_train_yolov8.py
```

8. After training, run custom inference:

```python
!python assignment_artifacts/inference_with_custom_feature.py \
  --model runs/detect/viad_yolov8n_5class/weights/best.pt \
  --source assignment_artifacts/viad_5class_500/images/test \
  --save
```

## Custom feature included

The assignment required at least one extra feature. The inference script supports:

- object counting
- detect only specific classes
- alert when no person is detected

Example:

```python
!python assignment_artifacts/inference_with_custom_feature.py \
  --model runs/detect/viad_yolov8n_5class/weights/best.pt \
  --source assignment_artifacts/viad_5class_500/images/test \
  --only-classes person car bus \
  --alert-no-person \
  --save
```

## What to submit

- code: this folder and scripts
- trained model file: `best.pt`
- demo screenshots or short video from prediction results
- short report: update `assignment_artifacts/report.md` with your final metrics and screenshots

## Optional FastAPI Wrapper

If you want to go beyond the base assignment, there is also an optional FastAPI wrapper around the trained model. This is a good way to present the project as a lightweight inference service.

FastAPI files:

- `fastapi_app.py`
- `requirements-fastapi.txt`
- `FASTAPI_USAGE.md`

The API supports:

- health check
- full detection output
- object counting
- class-filtered detection
- no-person alert responses
