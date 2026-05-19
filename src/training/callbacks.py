# src/training/callbacks.py
import os
import numpy as np

class TrainingMonitor:
    """Menggantikan peran callback Keras dalam custom loop."""
    def __init__(self, model_dir, patience=5):
        self.model_dir = model_dir
        self.patience = patience
        self.best_val_acc = 0.0
        self.wait = 0
        self.stopped_early = False

    def on_epoch_end(self, epoch, val_acc, model):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.wait = 0
            # Simpan model terbaik
            model.save(os.path.join(self.model_dir, 'best_model.keras'))
            print(f"  ✅ Best model saved (val_acc: {val_acc:.4f})")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_early = True
                print(f"  ⏹️ Early stopping triggered after {epoch+1} epochs.")
        return self.stopped_early