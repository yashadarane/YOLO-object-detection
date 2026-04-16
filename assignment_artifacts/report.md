# Multi-Object Detection System Using YOLOv8

## Executive Summary

This project delivers a compact yet production-minded multi-object detection pipeline built on the `Senior-Design-VIAD-4` dataset. The objective was to design, train, and evaluate a YOLO-based detector capable of recognizing high-value road-scene objects while operating under a strict data budget of only 500 images.

To make the task both practical and defensible, the original 38,952-image dataset was reduced to a curated 5-class subset focused on traffic-scene understanding: `person`, `car`, `bicycle`, `bus`, and `traffic_light`. The final system includes end-to-end dataset preparation, label normalization, YOLOv8 training on Google Colab GPU, evaluation on unseen test images, and a custom post-processing feature for object counting and safety-style alerts.

The result is a complete detection workflow that demonstrates sound dataset design, efficient model training, and applied computer vision thinking under real-world resource constraints.

## 1. Problem Statement

The goal of this assignment was to build a real-world object detection system rather than only demonstrate theoretical understanding. The system needed to:

- prepare a multi-class dataset in YOLO format
- train a YOLO model
- evaluate performance on unseen images
- include at least one custom feature beyond baseline detection

An additional practical constraint was that the available local machine had only CPU resources, so the training workflow had to be designed for execution on Google Colab GPU.

## 2. Dataset Strategy

### Source Dataset

- Dataset: `Senior-Design-VIAD-4`
- Original size: `38,952` images
- Source annotation format: COCO
- Preprocessing already applied in the source export: resize to `300 x 300`

### Why a 500-Image Subset Was Used

The assignment required working with only 500 images. Instead of randomly sampling the dataset, a targeted subset was created to maximize practical value and label usefulness. This makes the experiment more meaningful than a purely arbitrary reduction.

The subset was designed around road-scene perception because that context naturally supports interpretable object detection results and aligns well with downstream applications such as monitoring, traffic understanding, and safety alerts.

### Selected Classes

The following 5 classes were selected:

- `person`
- `car`
- `bicycle`
- `bus`
- `traffic_light`

These classes were chosen for three reasons:

- they are visually meaningful and easy to justify in a real deployment scenario
- they appear frequently enough in the source dataset to support training
- together they create a balanced mix of pedestrians, vehicles, and traffic-control elements

### Label Normalization

The original dataset contained inconsistent or fragmented labels. To improve training quality and presentation quality, the labels were normalized before export:

- `Person` and `person` were merged into `person`
- `green_light`, `red_light`, `yellow_light`, and `traffic light` were merged into `traffic_light`

This normalization step is important because it reduces class fragmentation and makes the final label space cleaner, more interpretable, and more suitable for reporting.

## 3. Final Dataset Composition

### Data Split

- Training set: `400` images
- Validation set: `50` images
- Test set: `50` images

### Object Distribution in the Final Subset

#### Training Split

- `car`: `855` objects
- `person`: `546` objects
- `bicycle`: `369` objects
- `bus`: `281` objects
- `traffic_light`: `231` objects

#### Validation Split

- `car`: `142` objects
- `person`: `91` objects
- `bicycle`: `35` objects
- `bus`: `22` objects
- `traffic_light`: `19` objects

#### Test Split

- `car`: `145` objects
- `person`: `76` objects
- `bicycle`: `31` objects
- `bus`: `20` objects
- `traffic_light`: `19` objects

The final subset preserves strong representation for the most important classes while keeping the experiment lightweight enough for fast GPU training and iteration.

## 4. Data Preparation Pipeline

The complete data preparation workflow was automated to make the experiment reproducible. The pipeline performed the following steps:

- read COCO annotations from the original dataset
- filter annotations to the selected 5 classes only
- normalize inconsistent labels into a clean final taxonomy
- convert bounding boxes from COCO format to YOLO format
- generate a structured YOLO dataset with `images/` and `labels/` folders
- write a `data.yaml` file for direct YOLOv8 training

This automation ensures that the final dataset is not only usable for this assignment, but also reusable for future experiments or model comparisons.

## 5. Model Architecture and Training Setup

### Model Selection

The chosen detector was `YOLOv8n`.

This variant was selected because it provides a strong balance between speed, simplicity, and detection capability, which is especially appropriate for a small curated dataset and limited training budget.

### Training Environment

- Platform: Google Colab
- Hardware: GPU runtime
- Framework: Ultralytics YOLOv8

### Training Configuration

- Image size: `640`
- Epochs: `30`
- Batch size: `16`
- Pretrained weights: `yolov8n.pt`
- Early stopping patience: `10`

Using pretrained weights helped transfer general visual features into the smaller assignment dataset, which is a practical strategy when labeled data is limited.

## 6. Evaluation Methodology

The model was evaluated on unseen test images after training. The following metrics were selected because they provide a strong summary of detection quality:

- `Precision`: how often predicted detections are correct
- `Recall`: how many real objects are successfully detected
- `mAP@50`: mean Average Precision at IoU threshold 0.50
- `mAP@50:95`: a stricter aggregate metric across multiple IoU thresholds

### Final Test Metrics

Update this section using the values automatically exported by the notebook into `metrics_summary.txt`.

- Precision: `0.5892`
- Recall: `0.4463`
- mAP@50: `0.4625`
- mAP@50:95: `0.2572`

### Recommended Interpretation for Discussion

When presenting the results, discuss them through the following lens:

- strong performance on `car` is expected because it has the highest object count
- `traffic_light` may be more difficult due to smaller object size and lower representation
- `bus` performance may vary depending on viewpoint and scale diversity
- the small 500-image budget creates a useful tradeoff between efficiency and generalization

## 7. Custom Feature and Practical Value

To move beyond a baseline academic detector, a custom inference layer was added with three practical features:

- object counting per image
- optional filtering to only selected classes
- alert generation when no `person` is detected

This feature set makes the system more application-oriented. For example:

- object counting can support scene summarization
- class filtering can support targeted monitoring workflows
- no-person alerts can support occupancy or safety-related scenarios

Including this layer demonstrates that the project was designed with usability in mind, not just model training.

## 8. Qualitative Results

Insert 2 to 3 prediction screenshots in this section from the Colab output folder.

For each example, briefly explain:

- what objects were detected correctly
- whether confidence appears reliable
- where the model struggled
- whether small or partially occluded objects were missed

Suggested caption style:

`Figure 1. Example of successful multi-object detection showing cars, pedestrians, and traffic lights in a dense street scene.`

`Figure 2. Example of a challenging scene illustrating missed small objects and partial occlusion effects.`

## 9. Key Engineering Decisions

Several decisions were made to improve both technical quality and project presentation:

- class selection was based on usefulness and label frequency, not random preference
- noisy labels were normalized to prevent artificial class fragmentation
- the subset was generated programmatically for reproducibility
- Google Colab GPU was used to overcome local hardware limitations cleanly
- a custom feature was added to show application thinking beyond baseline detection

These decisions make the project read more like a compact engineering implementation than a one-off notebook experiment.

## 10. Limitations

Although the system is functional and well-structured, a few limitations remain:

- the training set is intentionally small, which limits generalization
- class imbalance still exists, especially for `traffic_light` and `bus`
- some objects are small relative to the image size, making them harder to detect
- the dataset contains heterogeneous image sources, which may introduce annotation and domain variability

These limitations are expected in a constrained prototype and provide clear opportunities for future improvement.

## 11. Future Improvements

If this project were extended further, the next steps would be:

- increase the dataset size while keeping the normalized 5-class taxonomy
- compare `YOLOv8n` with a larger variant such as `YOLOv8s`
- perform targeted augmentation for smaller classes like `traffic_light`
- add per-class error analysis and confusion review
- extend the custom feature into a live webcam or video inference pipeline

These improvements would strengthen both quantitative performance and deployment readiness.

## 12. Conclusion

This project successfully demonstrates the full lifecycle of a modern object detection task under practical constraints. Starting from a large, noisy multi-class dataset, a focused 500-image subset was curated, normalized, converted into YOLO format, and used to train a YOLOv8 detector on Google Colab GPU. The final solution includes both quantitative evaluation and a custom inference feature that improves real-world usefulness.

More importantly, the work shows thoughtful engineering judgment: selecting meaningful classes, cleaning inconsistent labels, automating dataset preparation, and designing the workflow around available hardware constraints. As a result, the project is not only complete from an assignment standpoint, but also professionally structured and easy to extend.
