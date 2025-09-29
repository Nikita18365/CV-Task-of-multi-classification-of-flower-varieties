🌸 Multi-class Classification of Flower Varieties

This project demonstrates a complete pipeline for solving a multi-class image classification problem using deep learning. The dataset consists of flower images grouped by species. The goal is to train and evaluate convolutional neural networks (CNNs) to automatically recognize the type of flower from an image.

**📂 Project Overview:**
- 1. Data Preparation

Images are organized in folders by flower categories.

Custom data generators (CustomDataGenerator) and ImageDataGenerator are used to load, preprocess, and augment the images.

Augmentations include rotations, shifts, flips, zoom, brightness/contrast changes, and noise.

- 2. Model Architecture

A baseline CNN is implemented with multiple Conv2D + MaxPooling2D layers, followed by fully connected (Dense) layers.

The final layer uses softmax activation for multi-class classification.

A transfer learning approach with ResNet50 (pre-trained on ImageNet) is also explored.

- 3. Training & Validation

The model is trained using sparse_categorical_crossentropy loss and the Adam optimizer.

Training is performed with K-Fold cross-validation to ensure robust evaluation.

Metrics such as accuracy and loss are plotted for both training and validation sets.

- 4. Testing

After cross-validation, the model is evaluated on a completely unseen test dataset.

Final accuracy is reported, along with the potential to generate confusion matrices and classification reports.

**Tech Stack:**
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, scikit-learn
