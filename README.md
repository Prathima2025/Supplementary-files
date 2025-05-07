TITLE : 
Brain-Tumor-Detector
A deep learning-based system for brain tumor detection and classification using CNN with TensorFlow and Keras.

INTRODUCTION : 
Brain tumors represent one of the most life-threatening health conditions requiring early and accurate diagnosis. 
Traditional manual analysis of MRI scans is time-consuming and prone to error. 
Leveraging deep learning, particularly Convolutional Neural Networks (CNNs), allows for efficient and accurate detection of brain tumors from MRI scans. 
This project builds a CNN-based detection model using a publicly available brain MRI dataset from Kaggle, applying data augmentation and preprocessing to improve performance. 
The model achieves a test accuracy of 98.98%, highlighting its potential for real-world medical applications.

DESCRIPTIONS: 
This project implements a Convolutional Neural Network (CNN) from scratch to detect brain tumors from MRI images. 
The model is designed to perform binary classification (tumor vs. non-tumor) and achieves high accuracy with data augmentation and custom architecture, addressing overfitting and limited computational resources.

DATASET INFORMATION:
Source: Kaggle - Brain Tumor Classification
Classes:
yes: 155 MRI images with brain tumors
no: 98 MRI images without tumors
After Augmentation:
1085 positive images
980 negative images
Total: 2065 images (including originals)


CODE INFORMATION:
1. Data Augmentation
To counter dataset imbalance and improve generalization:
Techniques used: Rotation, Zoom, Flip, etc.
Resulting dataset: 1085 (tumor) + 980 (non-tumor) = 2065 images

2. Preprocessing
Crop brain-only region
Resize to (240, 240, 3)
Normalize pixel values to [0, 1]

3. Model Architecture
Custom CNN with the following layers:
Zero Padding
Conv2D (32 filters, 7x7)
Batch Normalization
ReLU Activation
MaxPooling (twice)
Flatten
Dense with Sigmoid (binary output)
Simpler architecture chosen over ResNet/VGG due to overfitting and limited hardware resources.

4. Performance
Accuracy on test set: 98.98%
F1 Score: 0.88
Outperformed other models (CNN-LSTM, Caps-VGGNet)


USAGE INSTRUCTIONS
Download Dataset
The brain MRI dataset can be downloaded from Kaggle: Brain Tumor Classification Dataset.
The dataset has two folders: yes (tumorous) and no (non-tumorous).

Data Preparation
Images are preprocessed by cropping the brain region, resizing to (240, 240, 3), and normalizing pixel values to [0, 1].
Augmented data (stored in the augmented data folder) increases total samples to 2065 for balanced training.

Train-Test Split
Data is split into 70% training, 15% validation, and 15% testing.

Load Trained Model
The best trained model file is named: cnn-parameters-improvement-23-0.91.model
Load it using:
python
Copy
Edit
from tensorflow.keras.models import load_model
best_model = load_model(filepath='models/cnn-parameters-improvement-23-0.91.model')

Run Code
Use the Jupyter/IPython notebooks provided in the repository to run training, testing, and analysis steps.


REQUIREMENTS:
The project requires the following Python libraries:
TensorFlow – for building and training the CNN model (tensorflow.keras used for model loading).
Keras – high-level neural networks API (integrated into TensorFlow).
NumPy – for numerical operations and handling image arrays.
Matplotlib – for plotting training and validation metrics (loss and accuracy plots).
OpenCV / PIL – (likely used) for image preprocessing like cropping, resizing, and normalization.
scikit-learn – for evaluation metrics like accuracy, F1-score, and train-test splitting.


IMPLEMENTATION STEPS:
Step 1: Install Required Libraries
bash
Copy
Edit
pip install tensorflow keras numpy matplotlib opencv-python

Step 2: Load and Preprocess the Dataset
python
Copy
Edit
import cv2, os
import numpy as np
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (240, 240))
    img = img / 255.0  # Normalize
    return img
# Example:
img = preprocess_image('yes/image(1).jpg')

Step 3: Split Dataset
python
Copy
Edit
from sklearn.model_selection import train_test_split

# X, y = np.array(images), np.array(labels)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

Step 4: Build CNN Model
python
Copy
Edit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, ZeroPadding2D
model = Sequential([
    ZeroPadding2D((2, 2), input_shape=(240, 240, 3)),
    Conv2D(32, (7, 7), strides=(1, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    Flatten(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Step 5: Train the Model
python
Copy
Edit
history = model.fit(X_train, y_train, epochs=24, validation_data=(X_val, y_val))

Step 6: Evaluate and Save Model
python
Copy
Edit
model.evaluate(X_test, y_test)
model.save('cnn-parameters-improvement-23-0.91.model')

Step 7: Load and Predict
python
Copy
Edit
from tensorflow.keras.models import load_model
model = load_model('cnn-parameters-improvement-23-0.91.model')
predictions = model.predict(X_test)


CITATION:
https://www.kaggle.com/datasets/rahimanshu/figshare-brain-tumor-classification).<br>



