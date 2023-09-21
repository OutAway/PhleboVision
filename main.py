print('\n\n IMPORTANDO BIBLIOTECAS')

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import pandas as pd
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

RUNS = 10 
model_name = 'test_all.h5'

print('\n\n COMPILANDO MODELO')
img_rows = 720
img_cols = 480   
img_channels = 3

img_dim = (img_cols, img_rows, img_channels)
img_input = Input(shape=img_dim)





def VGGupdated(input_tensor=None,classes=2):    
    
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
   
    model = Model(inputs = img_input, outputs = x, name='Eunice_VGG')


    return model



print('\n\n EXTRAINDO ARQUIVOS')
base_directory = 'images'
df_list = []
image_data = []
names = []

for root, dirs, files in os.walk(base_directory):
    for file in files:
        # Extract the relevant part of the path by removing the 'base_directory'
        relative_path = os.path.relpath(os.path.join(root, file), base_directory)

        # Split the relative path into folder and file parts
        folder_name, file_name = os.path.split(relative_path)

        # Append them as a tuple
        df_list.append((folder_name, file_name))
        image_data.append(os.path.join(root, file))  # Store the full path to the image
        names.append(folder_name)

phlebo_df = pd.DataFrame(data=df_list, columns=['species', 'file name'])



print('\n\n PROCESSANDO IMAGENS')
img_array = []
for img_path in image_data:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #  BGR 2 RGB
    img = cv2.resize(img, (img_rows,img_cols))  
    img = img.astype('float32') / 255.0
    img_array.append(img)
img_array = np.array(img_array)



print('\n\nCHECKPOINT 1') 
if img_array.shape is not None:
   print('Image array shape:', img_array.shape, ' CHECK')
print('Image array shape:', img_array.shape)
if phlebo_df.shape is not None:
   print('Dataframe dimensions:', phlebo_df.shape, ' CHECK')  
if names is not None:
   if names == []:
      print('ERRO: BLANK LIST')
   else:
      print('Total de classes:', len(set(names)), ' CHECK')
if image_data is not None:
   if image_data == []:
      print('ERRO: BLANK LIST')
   else:
      print('imagens carregadas: =',len(image_data), 'CHECK')
	
print('\n\n SETUP DO TREINAMENTO - pode demorar um pouco') 



y=phlebo_df['species'].values
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform(y)
y=y.reshape(-1,1)
onehotencoder = OneHotEncoder()  
x_val= onehotencoder.fit_transform(y).todense()
#x_val= x_val.reshape(-1,1)
#x_val= x_val.transpose()


img_array, x_val = shuffle(img_array, x_val, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(img_array, x_val, test_size=0.05, random_state=415)
print('FORMATOS DE ENTRADA')
print('EVAL X',x_val.shape)
print('TREINO X',train_x.shape)
print('TREINO Y',train_y.shape)
print('TEST X',test_x.shape)
print('TEST Y',test_y.shape)

# Create a TensorBoard callback
log_dir = f"logs/fit/{time.strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = VGGupdated(classes = 3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs = RUNS, batch_size = 32, verbose='1', callbacks= tensorboard_callback)  
loss, accuracy = model.evaluate(test_x, test_y)
model.save('model_name')

print("Test loss:", loss)
print("Test accuracy:", accuracy)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f"Model_Accuracy_{model_name}.pdf")
