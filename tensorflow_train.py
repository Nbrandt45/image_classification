import numpy as np
import os
import cv2
import random

"""
Attributions:
https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
https://www.tensorflow.org/tutorials/
"""

class invalidImageError(Exception): pass


# get directory for images
data_dir = r".\silverware_images"
types = ["fork_images", "spoon_images", "knife_images"]

# set size for the images
IMG_SIZE = 100

training_data = []

""" add the images used to train our model"""
def add_training_data():
    for t in types:
        type_index = types.index(t)
        # path to files of utensils
        path = os.path.join(data_dir, t)
        #
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                sized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([sized_array, type_index])
                #plt.imshow(sized_array, cmap="gray")
                #plt.show()
            except invalidImageError as e:
                print(e)


add_training_data()

# shuffle the data for easier learning
random.shuffle(training_data)


data = []
labels = []

for img_data, label in training_data:
    data.append(img_data)
    labels.append(label)

# shape by:
# IMG_SIZE, IMG_SIZE : size of image
# 1 : image is set to grayscale
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



# save features for data
np.save('data_features.npy', data)


# save features for labels
np.save('labels_list.npy', labels)



