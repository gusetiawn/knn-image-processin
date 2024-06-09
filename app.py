import os
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Konfigurasi upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk ekstraksi fitur warna
def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Fungsi untuk memuat dan memproses dataset
def load_dataset(data_path):
    images = []
    labels = []
    
    # Implementasi pembuatan dataset sederhana
    for label in ['mentah', 'matang', 'terlalu_matang']:
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                image = cv2.imread(os.path.join(label_path, file))
                images.append(image)
                labels.append(label)
    
    return images, labels

# Memuat dan melatih model
data_path = 'dataset'  # Pastikan Anda memiliki folder 'dataset' dengan subfolder 'mentah', 'matang', 'terlalu_matang'
images, labels = load_dataset(data_path)
features = [extract_color_features(img) for img in images]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menerima upload gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Memproses gambar
        image = cv2.imread(filename)
        features = extract_color_features(image)
        ripeness = knn.predict([features])[0]
        
        return jsonify({'ripeness': ripeness})
    else:
        return jsonify({'error': 'Format file tidak didukung'}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)