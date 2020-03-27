import os, shutil
import keras
from keras import layers
from keras import models
from keras.models import load_model
from keras.preprocessing import image #Library for preprocessing the image into a 4D tensor
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import re

float_formatter = lambda x: "%.4f" % x

BaseDir = 'C:\Git\snow-model\snow-model\Test'; # '/Users/trevor-trou/Documents/Git/snow-model/Evaluate'

snowPath = os.path.join(BaseDir, 'Snow')
noSnowPath = os.path.join(BaseDir, 'Non')

snowImageNames = os.listdir(snowPath)
noSnowImageNames = os.listdir(noSnowPath)

jpg = re.compile('.*.jpg')

for img in snowImageNames:
    if not jpg.match(img):
        snowImageNames.remove(img)

for img in noSnowImageNames:
    if not jpg.match(img):
        noSnowImageNames.remove(img)

model = load_model('downsampled_model_layers_removed.h5')

# Clear screen
print(chr(27) + "[2J")

print("Evaluating Snow Images...")
for i in range(0, len(snowImageNames)):
    imgPath = os.path.join(snowPath, snowImageNames[i])
    img = image.load_img(imgPath,target_size=(150,150))

    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /=255.

    #Plot the test image
    # plt.imshow(img_tensor[0])
    # plt.show()

    prediction = model.predict(img_tensor)
    print(float_formatter(prediction[0][0])+ "\t" + snowImageNames[i])

print("\n\n")
print("Evaluating No Snow Images...")
for i in range(0, len(noSnowImageNames)):
    imgPath = os.path.join(noSnowPath, noSnowImageNames[i])
    img = image.load_img(imgPath,target_size=(150,150))

    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /=255.

    #Plot the test image
    # plt.imshow(img_tensor[0])
    # plt.show()

    prediction = model.predict(img_tensor)
    print(float_formatter(prediction[0][0]) + "\t" + noSnowImageNames[i])