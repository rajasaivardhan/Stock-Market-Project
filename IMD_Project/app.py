import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, 'models')
STATIC_PLOTS = os.path.join(APP_DIR, 'static', 'plots')
UPLOAD_DIR = os.path.join(APP_DIR, 'uploads')
DATA_DIR = os.path.join(APP_DIR, 'GASF_images')

ALLOWED_EXTENSIONS = {'.npy', '.csv'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-key'

# Load model at startup
MODEL_PATH = os.path.join(MODELS_DIR, 'model_cnn.h5')
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None


def allowed_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


def load_scalers_from_meta():
    # Currently features are already normalized via tanh in generator; no scalers kept
    return None


def prepare_input_from_npy(file_path: str):
    arr = np.load(file_path, allow_pickle=True)
    # Accept either single sample image or dict format
    if isinstance(arr, np.ndarray) and arr.ndim in (2,3):
        # image only; build zero aux with correct dims by inferring from training meta
        # Try reading feature size from meta.json
        meta_path = os.path.join(DATA_DIR, 'meta.json')
        num_features = 16
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                num_features = int(meta.get('num_features', 16))
                img_h, img_w, img_c = meta.get('image_shape', [arr.shape[0], arr.shape[1], arr.shape[2] if arr.ndim==3 else 1])
        else:
            img_h, img_w = arr.shape[:2]
            img_c = arr.shape[2] if arr.ndim == 3 else 1
        img = arr.astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        aux = np.zeros((num_features,), dtype=np.float32)
        return img[np.newaxis, ...], aux[np.newaxis, ...]
    elif isinstance(arr, dict):
        img = arr.get('image').astype(np.float32)
        aux = arr.get('features').astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        return img[np.newaxis, ...], aux[np.newaxis, ...]
    else:
        raise ValueError('Unsupported .npy content. Expect image array or {image,features} dict.')


def predict_from_arrays(img_batch: np.ndarray, feat_batch: np.ndarray):
    if model is None:
        raise RuntimeError('Model not loaded. Please train the model first.')
    probs = model.predict([img_batch, feat_batch], verbose=0)
    preds = np.argmax(probs, axis=1)
    labels = ['Down/Flat', 'Up']
    return preds.tolist(), probs.tolist(), [labels[p] for p in preds]


@app.route('/')
def index():
    # Try to read metrics
    metrics_path = os.path.join(STATIC_PLOTS, 'metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return render_template('index.html', metrics=metrics)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        if filename.endswith('.npy'):
            Xi, Xf = prepare_input_from_npy(save_path)
        else:
            # CSV: reuse generator logic quickly by saving to tmp and computing last window
            from generate_gasf_images import compute_technical_indicators, build_dataset
            import pandas as pd
            df = pd.read_csv(save_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').set_index('Date')
            df = compute_technical_indicators(df)
            # Use defaults matching the trained model meta where possible
            meta_path = os.path.join(DATA_DIR, 'meta.json')
            img_shape = (64, 64, 3)
            channel_features = ['Close', 'RSI_14', 'MACD']
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    shape = meta.get('image_shape', img_shape)
                    if len(shape) == 3:
                        img_shape = tuple(shape)
            Xi_all, Xf_all, _ = build_dataset(df, window_size=60, horizon=1, step_size=1, image_size=img_shape[0], channel_features=channel_features)
            Xi, Xf = Xi_all[-1][np.newaxis, ...], Xf_all[-1][np.newaxis, ...]
        preds, probs, labels = predict_from_arrays(Xi, Xf)
        return jsonify({'pred': int(preds[0]), 'label': labels[0], 'probs': probs[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_page', methods=['POST'])
def predict_page():
    if 'file' not in request.files:
        flash('No file provided', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('Empty filename', 'danger')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Unsupported file type', 'danger')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    try:
        if filename.endswith('.npy'):
            Xi, Xf = prepare_input_from_npy(save_path)
        else:
            from generate_gasf_images import compute_technical_indicators, build_dataset
            import pandas as pd
            df = pd.read_csv(save_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').set_index('Date')
            df = compute_technical_indicators(df)
            meta_path = os.path.join(DATA_DIR, 'meta.json')
            img_shape = (64, 64, 3)
            channel_features = ['Close', 'RSI_14', 'MACD']
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    shape = meta.get('image_shape', img_shape)
                    if len(shape) == 3:
                        img_shape = tuple(shape)
            Xi_all, Xf_all, _ = build_dataset(df, window_size=60, horizon=1, step_size=1, image_size=img_shape[0], channel_features=channel_features)
            Xi, Xf = Xi_all[-1][np.newaxis, ...], Xf_all[-1][np.newaxis, ...]
        preds, probs, labels = predict_from_arrays(Xi, Xf)
        return render_template('index.html', prediction={'pred': preds[0], 'label': labels[0], 'probs': probs[0]})
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('index'))


@app.route('/metrics')
def metrics():
    metrics_path = os.path.join(STATIC_PLOTS, 'metrics.json')
    history_path = os.path.join(STATIC_PLOTS, 'history.json')
    metrics = None
    history = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    return render_template('metrics.html', metrics=metrics, history=history)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
