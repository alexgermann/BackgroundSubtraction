import os
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

videoName = '4_HSM_Combined'
baseDir = os.path.join('C:\Git\BackgroundSubtraction\Data\Frames', videoName)
modelName = videoName + '_downscale.h5'

TrainDir = os.path.join(baseDir, 'Train')
TestDir = os.path.join(baseDir, 'Test')
ValidationDir = os.path.join(baseDir, 'Validation')

dataGen = ImageDataGenerator(rescale = 1. / 255)
train_generator = dataGen.flow_from_directory(TrainDir, target_size = (150, 150), batch_size=32, class_mode='binary')
test_generator = dataGen.flow_from_directory(TestDir, target_size = (150, 150), batch_size=32, class_mode='binary')
validation_generator = dataGen.flow_from_directory(ValidationDir, target_size = (150, 150), batch_size=32, class_mode='binary')

# Examine output
# for data_batch, labels_batch in train_generator:
#     print("data batch shape: ", data_batch.shape)
#     print("labels batch shape: ", labels_batch.shape)
#     break

#Model Architecture
model = models.Sequential()

# First convolution layer
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))

# Second convolution layer
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))


# # Third convolution layer
# model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
# model.add(layers.MaxPool2D((2,2)))

# # Fourth convolution layer
# model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
# model.add(layers.MaxPool2D((2,2)))

#Fully Connected or Densely Connected Classifier Network
model.add(layers.Flatten()) # Flatten 3D output to 1D
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))

#Output layer with a single neuron since it is a binary class problem
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

#Configure the model for running
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(1e-04), metrics = ['acc'])

#Train the Model: Fit the model to the Train Data using a batch generator
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=5, validation_data = validation_generator, validation_steps = 50)

#Saving the Trained Model
model.save(modelName)



#Plotting the loss and accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs,acc,'b', label = 'Training Accuracy')
plt.plot(epochs,val_acc,'r',label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


plt.plot(epochs,loss,'b', label = 'Training loss')
plt.plot(epochs,val_loss,'r',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# Testing
test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50)
print("Test Accuracy = ", test_acc)
print("Test Loss = ", test_loss)