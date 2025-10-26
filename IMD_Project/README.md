# Stock GASF Predictor

A full-stack deep learning application that predicts next-day stock price movement by converting time series windows into Gramian Angular Summation Field (GASF) images and feeding them to a hybrid CNN that fuses image features with numeric technical factors. The model is adversarially fine-tuned with FGSM and deployed as a Flask web app.

## Features
- Data generation from yfinance (2018–2023), technical indicators, sliding windows
- Multi-channel GASF images + auxiliary numeric indicators
- Hybrid CNN (multi-scale convs + squeeze-excite) and optional light ResNet
- Adversarial training via FGSM
- Evaluation with accuracy, precision, recall, F1, confusion matrix and plots
- Flask UI with upload (npy/csv), API prediction, and metrics page

## Project Structure
```
IMD_Project/
├── GASF_images/
│   ├── X_images.npy
│   ├── X_features.npy
│   ├── y_labels.npy
│   └── meta.json
├── models/
│   ├── model_cnn.h5
│   └── saved_model/
├── static/
│   └── plots/
│       ├── confusion_matrix.png
│       ├── accuracy_curve.png
│       ├── pred_vs_actual.png
│       └── metrics.json
├── templates/
│   ├── index.html
│   └── metrics.html
├── uploads/
├── app.py
├── generate_gasf_images.py
├── model_zoo.py
├── train_cnn.py
├── adversarial_training.py
├── evaluate_model.py
└── requirements.txt
```

## Setup

1. Python 3.10+
2. Install dependencies:
```bash
pip install -r IMD_Project/requirements.txt
```

## Data Generation
Generate GASF images from yfinance data (AAPL 2018–2023 by default):
```bash
python IMD_Project/generate_gasf_images.py \
  --ticker AAPL --start 2018-01-01 --end 2023-12-31 \
  --window 60 --horizon 1 --step 1 --img_size 64 \
  --channels Close,RSI_14,MACD \
  --out_dir IMD_Project/GASF_images
```

## Train
Train the hybrid CNN (or use `--variant resnet`):
```bash
python IMD_Project/train_cnn.py \
  --data_dir IMD_Project/GASF_images \
  --plots_dir IMD_Project/static/plots \
  --models_dir IMD_Project/models \
  --variant hybrid --epochs 30 --batch_size 64
```

## Adversarial Fine-tuning (FGSM)
```bash
python IMD_Project/adversarial_training.py \
  --data_dir IMD_Project/GASF_images \
  --model_path IMD_Project/models/model_cnn.h5 \
  --eps 0.03 --adv_ratio 0.5 --epochs 10
```

## Evaluate
```bash
python IMD_Project/evaluate_model.py \
  --data_dir IMD_Project/GASF_images \
  --model_path IMD_Project/models/model_cnn.h5 \
  --plots_dir IMD_Project/static/plots
```

## Run Web App
```bash
python IMD_Project/app.py
```
Open `http://localhost:5000`.

- Upload a `.npy` containing either a single GASF image (H×W or H×W×C) or a dict with keys `image` and `features`.
- Or upload a raw `.csv` with `Date,Open,High,Low,Close,Volume` columns; the app will compute indicators and the last window for prediction.

## Notes
- Label: 1 if future close (t + horizon) > last close in window; else 0.
- Images are scaled to [-1, 1] per-window, numeric features are robustly normalized with tanh.
- Chronological splits avoid leakage. Class weighting combats imbalance.
- The hybrid model fuses multi-scale conv features with numeric indicators, with squeeze-excite attention, improving robustness and accuracy.
