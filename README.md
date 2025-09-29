🌸 Multi-class Classification of Flower Varieties

This project demonstrates a complete pipeline for solving a multi-class image classification problem using deep learning. The dataset consists of flower images grouped by species. The goal is to train and evaluate convolutional neural networks (CNNs) to automatically recognize the type of flower from an image.

**Project Overview**

- Data preparation

The images are organized into folders by color categories.

A custom data generator (CustomDataGenerator) and an automatic data generator via the ImageDataGenerator protocol are used to download, pre-process, and supplement images.

The program provides different types of uploading image data to a project (via import tarfile, tf.keras.utils.get_file, tf.keras.preprocessing.image_dataset_from_directory)

Augmentation includes rotations, shifts, flips, zooming, brightness/contrast changes, and noise.

- Architecture of the model

The basic CNN is implemented using multiple Conv2D + MaxPooling2D layers, followed by fully connected (dense) layers.

At the last level, softmax activation is used for multiclass classification by the number of targets.

An approach to learning using ResNet50 (pre-trained in ImageNet) is also being considered.

The pre-trained ResNet50 model performed poorly compared to the simple CNN model. The difference in accuracy was approximately 30%.

- Training and validation

The model is trained using sparse_categorical_crossentropy loss and the Adam optimizer.

The accuracy function was used as a quality metric.

Training is performed both with K-fold cross-validation to ensure reliable evaluation, and with training on the entire dataset.

Metrics such as accuracy and loss are displayed for both training and validation datasets.

- Testing

After cross-validation, the model is evaluated on a completely invisible test dataset.

The test set includes a random sample from flower_photos.tgz for which the class labels are known.

The final accuracy is reported, as well as the possibility of creating confusion matrices and classification reports.

**Tech Stack:**
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, scikit-learn
