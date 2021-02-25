import tensorflow as tf 
import numpy as np 
import os
from imutils import paths
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TRAIN_PATH = "data"

image_paths = list(paths.list_images(TRAIN_PATH))
data = []
labels = []

for image_path in image_paths:
	label = image_path.split(os.path.sep)[-2]
	
	image = cv2.imread(image_path)
	# HSV?
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (128, 128))

	data.append(image)
	labels.append(label)

data = np.array(data)
labels = np.array(labels)

# one-hot encoding on the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)
print(labels)

(x_train, x_test, y_train, y_test) = train_test_split(data, labels,
	test_size=0.2, stratify=labels, random_state=42)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip=True)

datagen.fit(x_train)
