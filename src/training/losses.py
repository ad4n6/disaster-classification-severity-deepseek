# src/training/losses.py
import tensorflow as tf

class OrdinalFocalLoss(tf.keras.losses.Loss):
    """
    Focal loss + penalty ordinal (selisih indeks kelas).
    y_true: one-hot, y_pred: softmax probabilities.
    """
    def __init__(self, gamma=2.0, alpha=0.25, delta=0.1, name='ordinal_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta

    def call(self, y_true, y_pred):
        # Cross-entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)  # p_t
        focal = self.alpha * tf.pow(1. - pt, self.gamma) * ce

        # Ordinal penalty
        y_pred_class = tf.argmax(y_pred, axis=1)
        y_true_class = tf.argmax(y_true, axis=1)
        error = tf.cast(tf.abs(y_pred_class - y_true_class), tf.float32)
        ordinal_penalty = self.delta * error

        return focal + ordinal_penalty