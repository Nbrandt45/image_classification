import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.keras.utils import to_categorical
import random
from keras.models import load_model
import matplotlib.pyplot as plt

"""
Attributions:
https://www.tensorflow.org/tutorials/keras/classification#evaluate_accuracy
"""


data_dir = r".\test_images"
types = ["fork_images", "spoon_images", "knife_images"]

categories = ["Fork", "Spoon", "Knife"]



def load_data(filepath):
    IMG_SIZE = 100
    test_data = []
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    #plt.imshow(new_array, cmap="gray")
    #plt.show()
    return new_array

model = tf.keras.models.load_model('image_recognition.h5')

path = os.path.join(data_dir, "fork.jpg")
#input_test_data = load_data(path)


#model.summary()


train_size = 80

data = np.load('data_features.npy')


label = np.load('labels_list.npy')

data = data/255.0

# generate 2d classification dataset
data, label = make_blobs(n_samples=90, centers=3, n_features=2, cluster_std=2, random_state=2)
# encode output variable
label = to_categorical(label)


# train on 10 images

testX = data[train_size:, :]
testy = label[train_size:]

labels = []
i = 0
while i < len(testy):
    labels.append(categories[np.argmax(testy[i])])
    i += 1

print(labels)


predictions = model.predict(testX)

_, test_acc = model.evaluate(testX, testy, verbose=0)

print('Test Accuraccy: %.3f' % (test_acc*100) + '%')

img_num = random.randint(0, 9)

print(f'Testing for {labels[img_num]}')

predict_img = np.argmax(predictions[img_num])
correct_label = np.argmax(testy[img_num])


print()

if predict_img == 0:
    print("Image identified as fork")
elif predict_img == 1:
    print("Image identified as spoon")
elif predict_img == 2:
    print("Image identified as knife")

print()

if predict_img == correct_label:
    print("Item was correctly identified")
else:
    print("Incorrect Identification")









