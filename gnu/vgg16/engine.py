import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
# from keras.layers import GlobalMaxPooling2D
# from keras.layers import GlobalAveragePooling2D
# from keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras import backend as K
# from keras.applications.imagenet_utils import decode_predictions
# from keras.applications.imagenet_utils import preprocess_input
## from keras.applications.imagenet_utils import _obtain_input_shape ## keras =< 2.2.0
# from keras.utils import get_source_inputs

def VGGupdated(input_tensor=None,classes=2):

    img_rows, img_cols = 300, 300   # by default size is 224,224
    img_channels = 3

    img_dim = (img_rows, img_cols, img_channels)

    img_input = Input(shape=img_dim)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.


    model = Model(inputs = img_input, outputs = x, name='VGGdemo')


    return model

import numpy as np
import pandas as pd

import os

root_dir = 'images'  # you need to define this
dataset_cls = os.listdir(root_dir)

class_types = os.listdir(root_dir)
print (class_types)  # what kinds of rooms are in this dataset
print("Classes: ", len(dataset_cls))
for entry in class_types:
    num_files = len(os.listdir(os.path.join(root_dir, entry)))
    print(f"{entry}: {num_files} entries")

entry_files = []  # I've changed the variable name to avoid confusion

for entry in class_types:
    # Get all the file names
    all_entry = os.listdir(os.path.join(root_dir, entry))

    # Add them to the list
    for file in all_entry:
        entry_files.append((entry, os.path.join(root_dir, entry, file)))
#print(entry_files)  # this variable 'rooms' is not defined anywhere


model = VGGupdated(classes = len(dataset_cls)) # bedroom and diningroom
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Build a dataframe
phlebo_df = pd.DataFrame(data=entry_files, columns=['species', 'file'])
print(phlebo_df.head())
print(phlebo_df.tail())


# Let's check how many samples for each category are present
print("Total number of rooms in the dataset: ", len(phlebo_df))

room_count = phlebo_df['species'].value_counts()

print("rooms in each category: ")
print(room_count)



import cv2
path = root_dir + '/'


im_size = 300

images = []
labels = []

for i in dataset_cls:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path) ]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

#print(images)
#print(labels)

images = np.array(images)

images = images.astype('float32') / 255.0
images.shape


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
y=phlebo_df['species'].values
#print(y[:5])

y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)

from sklearn.preprocessing import OneHotEncoder
y=y.reshape(-1,1)
onehotencoder = OneHotEncoder()  # No need to specify categorical_features
Y= onehotencoder.fit_transform(y)
Y.shape  #(40, 2)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#import tensorflow as tf


images, Y = shuffle(images, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

import tensorflow as tf
# from tensorflow.keras.utils import to_categorical

# flatten the labels if they're not already flat
train_y = train_y.toarray()
test_y = test_y.toarray()




# Convert only if they are in SparseTensor format
if isinstance(train_x, tf.SparseTensor):
    train_x = tf.sparse.to_dense(train_x)
if isinstance(train_y, tf.SparseTensor):
    train_y = tf.sparse.to_dense(train_y)
if isinstance(test_x, tf.SparseTensor):
    test_x = tf.sparse.to_dense(test_x)
if isinstance(test_y, tf.SparseTensor):
    test_y = tf.sparse.to_dense(test_y)

# inspect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

