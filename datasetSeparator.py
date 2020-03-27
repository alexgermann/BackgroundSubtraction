import os
import shutil
import random


# Must be run on each video twice, once for Leak and once for NoLeak
video = '4_HSM_Toggling'
category = 'Leak'
baseDir = os.path.join('C:\Git\BackgroundSubtraction\Data\Frames', video)
srcDir = os.path.join(baseDir, category)
trainDir = os.path.join(baseDir, 'Train', category) 
testDir = os.path.join(baseDir, 'Test', category)
validationDir = os.path.join(baseDir, 'Validation', category)


# Separate into 70% training and 30% test
for filename in os.listdir(srcDir):
  chance = random.randint(1, 101)
  if chance <= 70:
    os.rename(os.path.join(srcDir, filename), os.path.join(trainDir, filename))
  else:
    os.rename(os.path.join(srcDir, filename), os.path.join(testDir, filename))


# Split 15% of training into validation
for filename in os.listdir(trainDir):
  chance = random.randint(1, 101)
  if chance <= 15:
    os.rename(os.path.join(trainDir, filename), os.path.join(validationDir, filename))
