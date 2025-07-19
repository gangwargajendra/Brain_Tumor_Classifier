import pickle
import os
import matplotlib.pyplot as plt

# Load saved training history
history_path = os.path.join(os.path.dirname(__file__), 'Brain_Tumor_Results', 'training_history.pkl')
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
history_path = os.path.join(os.path.dirname(__file__), 'Brain_Tumor_Results', 'history_plot.png')
plt.savefig(history_path)  # Optional: save as PNG
plt.show()
