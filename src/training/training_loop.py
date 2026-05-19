# src/training/training_loop.py
import tensorflow as tf
import os
import datetime
from .losses import OrdinalFocalLoss
from .callbacks import TrainingMonitor
import numpy as np

def train_model(model, train_ds, val_ds, config, config_loss, output_config):
    # --- Setup mixed precision (opsional) ---
    if config['mixed_precision']:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # --- Optimizer & loss ---
    # Fase 1: head saja
    lr1 = config['learning_rate_phase1']
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr1,
                                          weight_decay=config['weight_decay'])
    loss_fn = OrdinalFocalLoss(gamma=config_loss['gamma'],
                               alpha=config_loss['alpha'],
                               delta=config_loss['delta'])

    # --- TensorBoard ---
    log_dir = output_config['log_dir']
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time, 'train')
    val_log_dir = os.path.join(log_dir, current_time, 'val')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # --- Monitoring ---
    model_dir = output_config['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    monitor = TrainingMonitor(model_dir, patience=5)

    # --- Metrics ---
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    # --- Phase 1: Training head (backbone frozen) ---
    print("\n🔧 Phase 1: Training head...")
    for epoch in range(config['phase1_epochs']):
        print(f"\nEpoch {epoch+1}/{config['phase1_epochs']}")
        # Reset metrik
        train_acc_metric.reset_state()
        epoch_loss_avg = tf.keras.metrics.Mean()

        # Iterasi training
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            tf.clip_by_global_norm(grads, config['gradient_clip_norm'])
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss)
            train_acc_metric.update_state(y_batch, logits)

        train_acc = train_acc_metric.result().numpy()
        train_loss = epoch_loss_avg.result().numpy()

        # Validasi
        val_acc_metric.reset_state()
        val_loss_avg = tf.keras.metrics.Mean()
        for x_batch, y_batch in val_ds:
            val_logits = model(x_batch, training=False)
            val_loss = loss_fn(y_batch, val_logits)
            val_loss_avg.update_state(val_loss)
            val_acc_metric.update_state(y_batch, val_logits)
        val_acc = val_acc_metric.result().numpy()
        val_loss = val_loss_avg.result().numpy()

        # Logging
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('accuracy', train_acc, step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
            tf.summary.scalar('accuracy', val_acc, step=epoch)

        # Callback monitoring (simpan model terbaik)
        if monitor.on_epoch_end(epoch, val_acc, model):
            break  # early stopping

    # --- Phase 2: Fine-tuning (unfreeze sebagian backbone) ---
    print("\n🔓 Phase 2: Fine-tuning...")
    # Unfreeze backbone (semua layer bisa di-set trainable=True, atau sebagian atas saja)
    base_model = model.get_layer('efficientnetb1')  # pastikan nama layer backbone
    base_model.trainable = True
    # Bekukan layer BatchNormalization agar stabil
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Optimizer dengan learning rate lebih kecil
    lr2 = config['learning_rate_phase2']
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr2,
                                          weight_decay=config['weight_decay'])

    monitor.best_val_acc = 0.0  # reset early stopping untuk phase 2 (opsional)
    monitor.wait = 0
    monitor.stopped_early = False

    for epoch in range(config['phase2_epochs']):
        print(f"\nEpoch {epoch+1}/{config['phase2_epochs']}")
        train_acc_metric.reset_state()
        epoch_loss_avg.reset_state()

        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            tf.clip_by_global_norm(grads, config['gradient_clip_norm'])
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss)
            train_acc_metric.update_state(y_batch, logits)

        train_acc = train_acc_metric.result().numpy()
        train_loss = epoch_loss_avg.result().numpy()

        # Validasi
        val_acc_metric.reset_state()
        val_loss_avg.reset_state()
        for x_batch, y_batch in val_ds:
            val_logits = model(x_batch, training=False)
            val_loss = loss_fn(y_batch, val_logits)
            val_loss_avg.update_state(val_loss)
            val_acc_metric.update_state(y_batch, val_logits)
        val_acc = val_acc_metric.result().numpy()
        val_loss = val_loss_avg.result().numpy()

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('accuracy', train_acc, step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
            tf.summary.scalar('accuracy', val_acc, step=epoch)

        if monitor.on_epoch_end(epoch, val_acc, model):
            break

    # --- Simpan model final (dalam format .keras) ---
    final_path = os.path.join(model_dir, output_config['final_model_name'])
    model.save(final_path)
    print(f"\n✅ Training selesai. Model final disimpan di {final_path}")
    return model