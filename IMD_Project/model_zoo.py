from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def squeeze_excite_block(inputs: tf.Tensor, reduction_ratio: int = 16) -> tf.Tensor:
    channels = int(inputs.shape[-1])
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(max(channels // reduction_ratio, 1), activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([inputs, se])


def multiscale_conv_block(
    inputs: tf.Tensor,
    filters: int,
    kernel_sizes=(3, 5, 7),
    l2: float = 1e-5,
    use_se: bool = True,
    name: str | None = None,
) -> tf.Tensor:
    branches = []
    for k in kernel_sizes:
        x = layers.Conv2D(
            filters,
            kernel_size=k,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2),
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        branches.append(x)
    x = layers.Concatenate(name=None if name is None else f"{name}_concat")(branches)
    if use_se:
        x = squeeze_excite_block(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    return x


def residual_block(inputs: tf.Tensor, filters: int, stride: int = 1, l2: float = 1e-5) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False,
                                 kernel_regularizer=regularizers.l2(l2))(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_gasf_hybrid_cnn(
    image_shape: Tuple[int, int, int],
    num_aux_features: int,
    num_classes: int = 2,
    l2: float = 1e-5,
) -> tf.keras.Model:
    # Image branch: multiscale convs with channel attention
    img_in = layers.Input(shape=image_shape, name='image_input')
    x = multiscale_conv_block(img_in, filters=24, kernel_sizes=(3, 5, 7), l2=l2)
    x = multiscale_conv_block(x, filters=48, kernel_sizes=(3, 5), l2=l2)
    x = multiscale_conv_block(x, filters=96, kernel_sizes=(3,), l2=l2)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Auxiliary numeric features branch
    aux_in = layers.Input(shape=(num_aux_features,), name='aux_input')
    a = layers.LayerNormalization()(aux_in)
    a = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(a)
    a = layers.Dropout(0.2)(a)
    a = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2))(a)

    # Fuse
    f = layers.Concatenate()([x, a])
    f = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2))(f)
    f = layers.Dropout(0.3)(f)
    f = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(f)
    out = layers.Dense(num_classes, activation='softmax', name='output')(f)

    model = models.Model(inputs=[img_in, aux_in], outputs=out, name='GASFHybridCNN')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_light_resnet(
    image_shape: Tuple[int, int, int],
    num_aux_features: int,
    num_classes: int = 2,
    l2: float = 1e-5,
) -> tf.keras.Model:
    img_in = layers.Input(shape=image_shape, name='image_input')
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2))(img_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)

    x = layers.GlobalAveragePooling2D()(x)

    aux_in = layers.Input(shape=(num_aux_features,), name='aux_input')
    a = layers.LayerNormalization()(aux_in)
    a = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(a)

    f = layers.Concatenate()([x, a])
    f = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(f)
    out = layers.Dense(num_classes, activation='softmax', name='output')(f)

    model = models.Model(inputs=[img_in, aux_in], outputs=out, name='GASFResNetLite')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model(
    image_shape: Tuple[int, int, int],
    num_aux_features: int,
    num_classes: int = 2,
    variant: str = 'hybrid',
    l2: float = 1e-5,
) -> tf.keras.Model:
    if variant == 'resnet':
        return build_light_resnet(image_shape, num_aux_features, num_classes, l2)
    return build_gasf_hybrid_cnn(image_shape, num_aux_features, num_classes, l2)
