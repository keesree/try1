import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import img_to_array
from PIL import Image
import requests
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

from lxml import html
import requests
import numpy as np


import os
import os.path
import matplotlib.image as mpimg
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device
from pathlib import Path



mel_dir = os.path.join(r'C:\Users\mrsve\OneDrive\Desktop\Train_Test_Folder\train\melanoma')
no_mel_dir = os.path.join(r'C:\Users\mrsve\OneDrive\Desktop\Train_Test_Folder\train\no_melanoma')
mel_names = os.listdir(mel_dir)
no_mel_names = os.listdir(no_mel_dir)






train_datagen = image_dataset_from_directory(r'C:\Users\mrsve\OneDrive\Desktop\Train_Test_Folder\train',
                                                  image_size=(200,200),
                                                  subset='training',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)
test_datagen = image_dataset_from_directory(r'C:\Users\mrsve\OneDrive\Desktop\Train_Test_Folder\test',
                                                  image_size=(200,200),
                                                  subset='validation',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)

model = tf.keras.models.Sequential([
	layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
	layers.MaxPooling2D(2, 2),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D(2, 2),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D(2, 2),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D(2, 2),

	layers.Flatten(),
	layers.Dense(512, activation='relu'),
	layers.BatchNormalization(),
	layers.Dense(512, activation='relu'),
	layers.Dropout(0.1),
	layers.BatchNormalization(),
	layers.Dense(512, activation='relu'),
	layers.Dropout(0.2),
	layers.BatchNormalization(),   
	layers.Dense(1, activation='sigmoid')
])

model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
)

model.fit(train_datagen,
		epochs=1,
		validation_data=test_datagen)


#plt.show() 


url = 'https://slp-fall-2023.s3.us-west-1.amazonaws.com/try1/melanoma.jpg'
im = Image.open(requests.get(url, stream=True).raw)

 
#For show image

im = im.resize((200, 200))
im = img_to_array(im)
test_image = np.expand_dims(im,axis=0)


# Result array
#im = r'C:\Users\mrsve\OneDrive\Desktop\Train_Test_Folder\melanoma.jpg'
result = model.predict(test_image)

#Mapping result array with the main name list
i=0
def getResult():
	if(result>=0.5):
		print("No melanoma")
		print(result)
	else:
		print("Melanoma")
		print(result)

getResult()