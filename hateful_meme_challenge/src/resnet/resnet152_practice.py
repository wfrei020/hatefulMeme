# based on https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/
#
#PLEASE SEE JUNYPER NOTEBOOK, THE TRAINING CODE THERE IS COMPLETE
#
############################################################################################################
######################################## RESNET TRAINING MODULE ############################################
############################################################################################################
import matplotlib
matplotlib.use("Agg")

import resnet152_config as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet152
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
############################################################################################################
#############################################IMAGE PREPROCESSING ###########################################
############################################################################################################
# let me load the data in a different manner
# image_path = "/Users/andrei/Ryerson/neuralNet_EE8204/codeTests/resnet/hateful_nothateful_images/validation/hateful/95640.png"
# image = load_img(image_path, target_size=(224, 224))
# input_arr = img_to_array(image)
# #input_arr = np.array([input_arr])
# print(input_arr.shape)
# exit()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

print("initializing the training data augmentation object")
# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")
# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
print("initializing the val data augmentation object")
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean


# initialize the training generator
print("initializing the training generator")
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BS)
# by the looks of it the train gnerator returns a dictinaryiterator [x,y] whre x is the batch and
# y is the labels
# print(len(trainGen.next()[1]))
# initialize the validation generator
print("initializing the val generator")
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BS)
# initialize the testing generator
print("initializing the testing generator")
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=1)

############################################################################################################
################################################### MODEL ##################################################
############################################################################################################

# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
print("[INFO] preparing model...")
baseModel = ResNet152(weights="imagenet", include_top=False,
                      input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)

headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)

headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)  # this is my softmax layer for output
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process

for layer in baseModel.layers:
    layer.trainable = False

    # compile the model
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

############################################################################################################
################################################# TRAINING #################################################
############################################################################################################
# train the model
print("[INFO] training model...")
H = model.fit(
    x=trainGen,
    steps_per_epoch=totalTrain // config.BS,
    validation_data=valGen,
    validation_steps=totalVal // config.BS,
    epochs=config.NUM_EPOCHS)

# print(model.summary())
model.save(config.MODEL_PATH, save_format="h5")
print("finished saving model")
exit()
# below is used to fo some testing
############################################################################################################
################################################### TESTING ################################################
############################################################################################################
#
# PLEASE SEE FUSE_BERT_RESNET.PY
#


# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
#predIdxs = model.predict(x=testGen, steps=(totalTest // config.BS) + 1)
# model.pop()
# model.make_test_function()
#test_loss, test_accuracy, test_prediction, test_catAccur = model.evaluate(x=testGen, steps=totalTest, callbacks=[CustomCallback(), LossAndErrorPrintingCallback()])
test_proba = model.predict(x=testGen, steps=totalTest)
# test_prob same as bert, it is [probaility of 0 and probaility of 1] based on batch of validation i need to train my model for restnet and so on
print(test_proba)


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(test_proba, axis=1)
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))
# serialize the model to disk
print("[INFO] saving model...")
model.save(config.MODEL_PATH, save_format="h5")

# plot the training loss and accuracy
