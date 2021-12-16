import glob
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import SGD  # stochasitic gradient descent optimizer
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_num_files(path):  # Get #files in folder and all subfolders
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


def get_num_subfolders(path):  # Get #subfolders directly below the folder in path
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])


# Define image generators that will variations of image with the image rotated slightly,
# shifted up, down, left, or right, sheared, zoomed in, or flipped horizontally on the
# vertical axis (ie. person looking to the left ends up looking to the right)
def create_img_generator(): # to improve translational invariance
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

Image_width, Image_height = 299, 299
Training_Epochs = 2
Batch_Size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'
num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)
num_epoch = Training_Epochs
batch_size = Batch_Size

train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

#   Connect the image generator to a folder contains the source images the image generator alters.
#   Training image generator
train_generator = train_image_gen.flow_from_directory(
  train_dir,
  target_size=(Image_width, Image_height),
  batch_size=batch_size,
  seed = 42    #set seed for reproducability
)

#   Validation image generator
validation_generator = test_image_gen.flow_from_directory(
  validate_dir,
  target_size=(Image_width, Image_height),
  batch_size=batch_size,
  seed=42       #set seed for reproducability
)

# Define trainable model which links input from the Inception V3 base model to the new classification prediction layers
model = load_model('inceptionv3-transfer-learning.model')

print('\nFine tuning existing model')
#   Freeze
Layers_To_Freeze = 172
for layer in model.layers[:Layers_To_Freeze]:
  layer.trainable = False
for layer in model.layers[Layers_To_Freeze:]:
  layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the Fine-tuning model to the data from the generators.
# By using generators we can ask continue to request sample images and
# the generators will pull images from the training or validation folders, alter then slightly, and
# pass the images back
history_fine_tune = model.fit_generator(
  train_generator,
  steps_per_epoch = num_train_samples // batch_size,
  epochs=num_epoch,
  validation_data=validation_generator,
  validation_steps = num_validate_samples // batch_size,
    class_weight='auto')

# Save fine tuned model
model.save('inceptionv3-fine-tune.model')