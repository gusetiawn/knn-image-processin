import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuration for upload folders
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
HISTOGRAM_FOLDER = 'histograms'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['HISTOGRAM_FOLDER'] = HISTOGRAM_FOLDER

# Function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract color features and save the processed image and histogram
def extract_color_features(image, processed_filename, histogram_filename):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Save the processed HSV image
    cv2.imwrite(processed_filename, hsv)

    # Create and save the histogram image
    hist_img = np.zeros((300, 300, 3), dtype=np.uint8)
    bin_w = int(round(300 / 512))

    for i in range(1, 512):
        cv2.line(hist_img, (bin_w * (i - 1), 300 - int(hist[i - 1] * 300)),
                 (bin_w * i, 300 - int(hist[i] * 300)), (255, 255, 255), thickness=2)

    cv2.imwrite(histogram_filename, hist_img)

    # Calculate mean hue, saturation, and value
    mean_hue = np.mean(hsv[:, :, 0])
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])

    return hist, mean_hue, mean_saturation, mean_value

# Function to load and process the dataset
def load_dataset(data_path):
    images = []
    labels = []

    for label in ['mentah', 'matang', 'terlalu_matang']:
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                image = cv2.imread(os.path.join(label_path, file))
                images.append(image)
                labels.append(label)

    return images, labels

# Load and train the model
data_path = 'dataset'
images, labels = load_dataset(data_path)
features = [extract_color_features(img, os.path.join(app.config['PROCESSED_FOLDER'], f'{i}.jpg'),
                                   os.path.join(app.config['HISTOGRAM_FOLDER'], f'{i}.png'))[0] for i, img in enumerate(images)]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if file and allowed_file(file.filename):
        upload_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_filename)

        # Process the image
        image = cv2.imread(upload_filename)
        processed_filename = os.path.join(app.config['PROCESSED_FOLDER'], file.filename)
        histogram_filename = os.path.join(app.config['HISTOGRAM_FOLDER'], file.filename.rsplit('.', 1)[0] + '.png')
        features, mean_hue, mean_saturation, mean_value = extract_color_features(image, processed_filename, histogram_filename)
        ripeness = knn.predict([features])[0]

        return jsonify({
            'ripeness': ripeness,
            'image_url': f'/processed/{file.filename}',
            'histogram_url': f'/histograms/{file.filename.rsplit(".", 1)[0]}.png',
            'histogram_data': {
                'hue_mean': mean_hue,
                'saturation_mean': mean_saturation,
                'value_mean': mean_value
            }
        })
    else:
        return jsonify({'error': 'Format file tidak didukung'}), 400

# Route to access processed images
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Route to access histogram images
@app.route('/histograms/<filename>')
def histogram_file(filename):
    return send_from_directory(app.config['HISTOGRAM_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    if not os.path.exists(HISTOGRAM_FOLDER):
        os.makedirs(HISTOGRAM_FOLDER)
    app.run(debug=True)
