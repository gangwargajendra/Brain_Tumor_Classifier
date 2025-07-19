import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset and model paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'brain_tumor_dataset')
test_dir = os.path.join(DATASET_DIR, 'Testing')

# Step 1: Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'Brain_Tumor_Results', 'best_brain_tumor_model.keras')
model = load_model(model_path)
print("✅ Loaded model: best_brain_tumor_model.keras")

# Step 2: Load the class label order used during training
class_labels_path = os.path.join(os.path.dirname(__file__), 'Brain_Tumor_Results', 'class_labels.npy')
class_labels = np.load(class_labels_path, allow_pickle=True)
print("✅ Loaded class labels:", class_labels)

# Step 3: Build test DataFrame (load up to 1300 images)
classes = []
file_paths = []
count = 0
max_images = 1200

for label in os.listdir(test_dir):
    print(label)
    label_path = os.path.join(test_dir, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if count >= max_images:
            break
        if file.lower().endswith('.jpg'):
            file_paths.append(os.path.join(label_path, file))
            classes.append(label)
            count += 1
    if count >= max_images:
        break

eval_df = pd.DataFrame({
    'file_paths': file_paths,
    'class': classes
})

# Step 4: ImageDataGenerator with consistent label mapping
gen = ImageDataGenerator()

print(class_labels)

eval_gen = gen.flow_from_dataframe(
    eval_df,
    x_col='file_paths',
    y_col='class',
    target_size=(244, 244),
    color_mode='rgb',
    class_mode='categorical',
    classes=class_labels.tolist(),  # ✅ Important: forces same label order
    shuffle=False,
    batch_size=8
)

# Step 5: Evaluate
loss, accuracy = model.evaluate(eval_gen)
print(f"\n✅ Validation Accuracy on {count} images: {accuracy * 100:.2f}%")
