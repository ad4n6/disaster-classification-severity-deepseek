# src/training/data_loader.py
import tensorflow as tf

def preprocess_efficientnet(image, label, image_size):
    """Resize dan preprocessing khusus EfficientNet."""
    image = tf.image.resize(image, image_size)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

def augment(image, label, config_aug):
    """Augmentasi ringan, bisa diperluas."""
    if config_aug.get("random_flip") == "horizontal":
        image = tf.image.random_flip_left_right(image)
    # Rotasi ±0.1*2π = ±36 derajat
    rot_factor = config_aug.get("random_rotation", 0.1)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Jika ingin continuous rotation, bisa pakai tfa.image.rotate
    # Zoom (crop + resize) - implementasi sederhana
    if config_aug.get("random_zoom", 0) > 0:
        scale = tf.random.uniform([], 1 - config_aug["random_zoom"], 1 + config_aug["random_zoom"])
        new_h = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * scale, tf.int32)
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(image, tf.shape(image)[0], tf.shape(image)[1])
    # Contrast
    if config_aug.get("random_contrast", 0) > 0:
        image = tf.image.random_contrast(image, lower=1-config_aug["random_contrast"],
                                        upper=1+config_aug["random_contrast"])
    return image, label

def create_datasets(data_dir, image_size, batch_size, config_aug, subset="training"):
    """
    Asumsi struktur data_dir:
        data_dir/
            ringan/
            sedang/
            berat/
    """
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset=subset,
        seed=42,
        image_size=image_size,    # resize langsung
        batch_size=batch_size,
        label_mode='categorical', # one-hot
        color_mode='rgb'
    )
    
    # Preprocessing EfficientNet (normalisasi)
    ds = ds.map(lambda x, y: preprocess_efficientnet(x, y, image_size))
    
    if subset == "training":
        # Cache setelah preprocessing (opsional, jika memori cukup)
        ds = ds.cache()
        ds = ds.shuffle(1000)
        # Augmentasi
        ds = ds.map(lambda x, y: augment(x, y, config_aug))
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds