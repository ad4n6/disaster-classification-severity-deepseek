# src/training/model.py
import tensorflow as tf

def build_model(input_shape=(240, 240, 3), num_classes=3, config_model=None):
    # Backbone
    base = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base.trainable = False  # fase 1

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = base(inputs, training=False)   # inference mode untuk BN
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    dropout = config_model.get('dropout', 0.35) if config_model else 0.35
    dense_units = config_model.get('dense_units', 256) if config_model else 256
    l2_reg = config_model.get('l2_reg', 1e-4) if config_model else 1e-4

    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation='swish',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.Dropout(dropout * 0.6)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs, outputs, name='EfficientNet_BuildingDamage')
    return model