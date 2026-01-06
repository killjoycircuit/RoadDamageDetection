# Road Damage Detection using YOLOv8

This project focuses on detecting and classifying different types of road surface damages from images using the YOLOv8 object detection model. The model predicts both the **location (bounding boxes)** and **type of damage** present in road images.

---
***All prediction files are stored in a `predictions/` folder and compressed into `submission.zip` for final submission.***

## Project Overview

Road infrastructure monitoring is crucial for maintenance and safety. In this project, we train a YOLOv8-based object detection model to identify multiple types of road damage such as cracks and potholes from images.

The final trained model is used to generate predictions on an unseen test dataset for hackathon evaluation.

---

##  Model Architecture

- **Model**: YOLOv8 (Ultralytics)
- **Variant Used**: `yolov8m` (medium)
- **Task**: Object Detection
- **Framework**: PyTorch + Ultralytics YOLO

YOLOv8 is a single stage detector that predicts bounding boxes and class probabilities in one forward pass, making it fast and suitable for real time applications.

---


### Class Mapping
| Class ID | Damage Type |
|--------|-------------|
| 0 | Longitudinal Crack |
| 1 | Lateral Crack |
| 2 | Alligator Crack |
| 3 | Pothole |
| 4 | Other Damage |

---

![001215](https://github.com/user-attachments/assets/ba90f71f-b979-4533-b46c-4eb5ede8bc8a)
![001264](https://github.com/user-attachments/assets/15a8013e-17cb-4f07-a87f-ec7f5221fc0a)
![001258](https://github.com/user-attachments/assets/ba01006d-ebd9-4c33-a7e8-ae30123ad24d)
![001267](https://github.com/user-attachments/assets/f34a5319-f9cc-4fd8-a39e-d8f8ec236740)
![001239](https://github.com/user-attachments/assets/8cc117b2-2c58-4384-b4a8-0d106724fca6)



## 🏋️ Training Details

- **Image Size**: 640 × 640
- **Epochs**: 50
- **Batch Size**: Adjusted based on GPU memory
- **Optimizer**: Default YOLOv8 optimizer
- **Losses**:
  - Box loss
  - Classification loss
  - Distribution Focal Loss (DFL)

The best model checkpoint (`best.pt`) is automatically selected based on **validation mAP@0.5:0.95**.

---

## 📊 Evaluation Metrics

We use **Mean Average Precision (mAP)** for evaluation:

- **mAP@0.5**: Measures detection quality at IoU ≥ 0.5
- **mAP@0.5:0.95**: Averaged over IoU thresholds from 0.5 to 0.95 (stricter, more reliable)

These metrics are standard for object detection benchmarks.

---

## 🔍 Inference & Prediction

Predictions are generated on the **test set** using the trained model.  
Each test image produces a corresponding `.txt` file in YOLO format:
`<class_id> <x_center> <y_center> <width> <height> <confidence_score>`


All prediction files are stored in a `predictions/` folder and compressed into `submission.zip` for final submission.

---

## 📁 Repository Contents

- `train_yolo.ipynb` – Training and inference notebook
- `rdd2022.yaml` – Dataset configuration
- `requirements.txt` – Python dependencies
- `.gitignore` – Ignored files and folders
- `predictions/` – Final test predictions (not tracked in Git)
- `runs/` – Training logs and outputs (ignored)

---

## ⚙️ Setup Instructions

Place the dataset in the **project root** as shown below:
```
yolov8_project/
├── randomized_dataset/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ ├── val/
│ │ ├── images/
│ │ └── labels/
│ └── test/
│ └── images/
```

The dataset path is defined in `rdd2022.yaml`:

```yaml
path: randomized_dataset
train: train/images
val: val/images
```


### 1. Create virtual environment
```bash
python -m venv yolov8env
source yolov8env/Scripts/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3.Train the model
```bash
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="rdd2022.yaml",
    epochs=50,
    imgsz=640,
    batch=2
)

```
