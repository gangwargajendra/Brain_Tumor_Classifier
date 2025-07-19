import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Dataset paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'brain_tumor_dataset')
train_dir = os.path.join(DATASET_DIR, 'Training')
test_dir = os.path.join(DATASET_DIR, 'Testing')

# Build train dataframe
classes = []
feature = []
folds = os.listdir(train_dir)
for fold in folds:
    fold_path = os.path.join(train_dir, fold)
    for file in os.listdir(fold_path):
        feature.append(os.path.join(fold_path, file))
        classes.append(fold)
train_df = pd.DataFrame({'file_paths': feature, 'class': classes})

# Build test dataframe
classes = []
feature = []
folds = os.listdir(test_dir)
for fold in folds:
    fold_path = os.path.join(test_dir, fold)
    for file in os.listdir(fold_path):
        feature.append(os.path.join(fold_path, file))
        classes.append(fold)
test_df = pd.DataFrame({'file_pathes': feature, 'class': classes})

# Split test into test and validation
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# # Data Augmentation
# gen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.3,
#     horizontal_flip=True,
#     brightness_range=[0.8, 1.2],
#     fill_mode='nearest'
# )
gen = ImageDataGenerator()

train_gen = gen.flow_from_dataframe(
    train_df,
    x_col='file_paths',
    y_col='class',
    class_mode='categorical',
    target_size=(244, 244),
    color_mode='rgb',
    batch_size=16
)

test_gen = gen.flow_from_dataframe(
    test_df,
    x_col='file_pathes',
    y_col='class',
    class_mode='categorical',
    target_size=(244, 244),
    color_mode='rgb',
    batch_size=8
)

valid_gen = gen.flow_from_dataframe(
    valid_df,
    x_col='file_pathes',
    y_col='class',
    class_mode='categorical',
    target_size=(244, 244),
    color_mode='rgb',
    batch_size=8
)

# CNN Model (based on claimed_model.txt)
model = Sequential([
    Conv2D(256, (3, 3), activation='relu', input_shape=(244, 244, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes
])

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy')
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, monitor='val_loss')
checkpoint = ModelCheckpoint(
    'best_brain_tumor_model.keras',  # Native Keras format
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Evaluate
print("Validation Accuracy:", model.evaluate(valid_gen)[1])
print("Test Accuracy:", model.evaluate(test_gen)[1])

# Save class labels for future use
class_labels = list(train_gen.class_indices.keys())
np.save('class_labels.npy', class_labels)

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)