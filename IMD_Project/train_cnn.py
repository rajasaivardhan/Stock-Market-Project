import os
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from model_zoo import build_model


def load_dataset(data_dir: str):
    X_images = np.load(os.path.join(data_dir, 'X_images.npy'))
    X_features = np.load(os.path.join(data_dir, 'X_features.npy'))
    y = np.load(os.path.join(data_dir, 'y_labels.npy'))
    meta_path = os.path.join(data_dir, 'meta.json')
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    return X_images, X_features, y, meta


def chronological_split(Xi, Xf, y, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    Xi_train, Xf_train, y_train = Xi[:n_train], Xf[:n_train], y[:n_train]
    Xi_val, Xf_val, y_val = Xi[n_train:n_train+n_val], Xf[n_train:n_train+n_val], y[n_train:n_train+n_val]
    Xi_test, Xf_test, y_test = Xi[n_train+n_val:], Xf[n_train+n_val:], y[n_train+n_val:]
    return (Xi_train, Xf_train, y_train), (Xi_val, Xf_val, y_val), (Xi_test, Xf_test, y_test)


def plot_history(history, out_path: str):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc, label='train acc')
    plt.plot(val_acc, label='val acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.title('Loss')
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    data_dir = args.data_dir
    plots_dir = args.plots_dir
    models_dir = args.models_dir
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    Xi, Xf, y, meta = load_dataset(data_dir)
    image_shape = Xi.shape[1:]
    num_features = Xf.shape[1]

    (Xi_tr, Xf_tr, y_tr), (Xi_va, Xf_va, y_va), (Xi_te, Xf_te, y_te) = chronological_split(
        Xi, Xf, y, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0,1]),
        y=y_tr
    )
    class_weight = {0: class_weights[0], 1: class_weights[1]}

    model = build_model(image_shape=image_shape, num_aux_features=num_features, variant=args.variant)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    history = model.fit(
        x=[Xi_tr, Xf_tr],
        y=y_tr,
        validation_data=([Xi_va, Xf_va], y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=2,
        callbacks=callbacks
    )

    # Save training history and plot
    history_path = os.path.join(plots_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    plot_history(history, os.path.join(plots_dir, 'accuracy_curve.png'))

    # Save model in both formats
    h5_path = os.path.join(models_dir, 'model_cnn.h5')
    model.save(h5_path)
    saved_model_dir = os.path.join(models_dir, 'saved_model')
    # Keras 3: export SavedModel for TF Serving/TFLite
    model.export(saved_model_dir)

    # Evaluate on test set for quick feedback
    test_results = model.evaluate([Xi_te, Xf_te], y_te, verbose=0)
    metrics_quick = {m: float(v) for m, v in zip(model.metrics_names, test_results)}
    with open(os.path.join(plots_dir, 'quick_test_metrics.json'), 'w') as f:
        json.dump(metrics_quick, f, indent=2)

    print('Saved model to:', h5_path)
    print('Quick test metrics:', metrics_quick)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='IMD_Project/GASF_images')
    parser.add_argument('--plots_dir', type=str, default='IMD_Project/static/plots')
    parser.add_argument('--models_dir', type=str, default='IMD_Project/models')
    parser.add_argument('--variant', type=str, default='hybrid', choices=['hybrid', 'resnet'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    args = parser.parse_args()
    main(args)
