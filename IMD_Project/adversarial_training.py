import os
import argparse
import json
import numpy as np
import tensorflow as tf


def load_dataset(data_dir: str):
    Xi = np.load(os.path.join(data_dir, 'X_images.npy'))
    Xf = np.load(os.path.join(data_dir, 'X_features.npy'))
    y = np.load(os.path.join(data_dir, 'y_labels.npy'))
    return Xi, Xf, y


def chronological_split(Xi, Xf, y, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        (Xi[:n_train], Xf[:n_train], y[:n_train]),
        (Xi[n_train:n_train+n_val], Xf[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        (Xi[n_train+n_val:], Xf[n_train+n_val:], y[n_train+n_val:])
    )


def fgsm_attack(model: tf.keras.Model, images: np.ndarray, features: np.ndarray, labels: np.ndarray, epsilon: float = 0.03, batch_size: int = 128):
    adv_images = np.empty_like(images)
    dataset = tf.data.Dataset.from_tensor_slices((images, features, labels)).batch(batch_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for step, (xi, xf, yi) in enumerate(dataset):
        xi = tf.cast(xi, tf.float32)
        xf = tf.cast(xf, tf.float32)
        yi = tf.cast(yi, tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(xi)
            preds = model([xi, xf], training=False)
            loss = loss_fn(yi, preds)
        grad = tape.gradient(loss, xi)
        signed_grad = tf.sign(grad)
        adv_xi = xi + epsilon * signed_grad
        adv_xi = tf.clip_by_value(adv_xi, -1.0, 1.0)
        adv_images[step*batch_size: step*batch_size + xi.shape[0]] = adv_xi.numpy()
    return adv_images


def main(args):
    Xi, Xf, y = load_dataset(args.data_dir)
    (Xi_tr, Xf_tr, y_tr), (Xi_va, Xf_va, y_va), (Xi_te, Xf_te, y_te) = chronological_split(Xi, Xf, y, args.train_ratio, args.val_ratio)

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    # Generate adversarial examples on current model
    adv_Xi_tr = fgsm_attack(model, Xi_tr, Xf_tr, y_tr, epsilon=args.eps, batch_size=args.batch_size)

    # Mix clean and adversarial samples
    ratio = args.adv_ratio
    n_adv = int(len(adv_Xi_tr) * ratio)
    idx = np.random.permutation(len(adv_Xi_tr))
    adv_idx = idx[:n_adv]

    Xi_tr_mix = Xi_tr.copy()
    Xi_tr_mix[adv_idx] = adv_Xi_tr[adv_idx]

    # Retrain briefly
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    model.fit(
        x=[Xi_tr_mix, Xf_tr], y=y_tr,
        validation_data=([Xi_va, Xf_va], y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks
    )

    # Save updated model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    saved_model_dir = os.path.join(os.path.dirname(args.model_path), 'saved_model')
    model.save(saved_model_dir)

    # Evaluate quick
    test_metrics = model.evaluate([Xi_te, Xf_te], y_te, verbose=0)
    metrics_quick = {m: float(v) for m, v in zip(model.metrics_names, test_metrics)}
    with open(os.path.join(args.plots_dir, 'quick_test_metrics_adv.json'), 'w') as f:
        json.dump(metrics_quick, f, indent=2)
    print('Adversarially trained model saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='IMD_Project/GASF_images')
    parser.add_argument('--model_path', type=str, default='IMD_Project/models/model_cnn.h5')
    parser.add_argument('--plots_dir', type=str, default='IMD_Project/static/plots')
    parser.add_argument('--eps', type=float, default=0.03)
    parser.add_argument('--adv_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    args = parser.parse_args()
    main(args)
