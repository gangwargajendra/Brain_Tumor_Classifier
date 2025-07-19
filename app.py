import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load model and class labels
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Brain_Tumor_Results', 'best_brain_tumor_model.keras')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'Brain_Tumor_Results', 'class_labels.npy')
model = load_model(MODEL_PATH)
class_labels = np.load(LABELS_PATH, allow_pickle=True)

def predict_tumor(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((244, 244))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    return class_labels[idx], float(np.max(preds[0]))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            label, conf = predict_tumor(filepath)
            prediction = label
            confidence = f"{conf*100:.2f}%"
            os.remove(filepath)
    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)