import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.util import lazy_loader

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    labels = list()
    images = list()




    #determine to proper way to address the root directory based on the operating system.
    print(" #determine to proper way to address the root directory based on the operating system.")
    rootdir = "."
    pathseperator = os.path.join(" ", " ").strip()
    rootdir = rootdir + pathseperator
    rootdir = rootdir + data_dir
    #print(rootdir) 

    catagoryNumber=""
    #Iterate over all the sub-directories within the root and create a list of all directory paths and the filenames that it contains
    print("#Iterate over all the sub-directories within the root and create a list of all directory paths and the filenames that it contains")
    for imagesubdir, imagedirs, imagefileslist in os.walk(rootdir):
        print("loading image from directory = ",imagesubdir)
        #Iterate over all directories and read each file in it
        for imagefile in imagefileslist:

            # extract the current directory name from it's path and assign it to categoryNumber variable
            catagoryNumber=imagesubdir.strip(rootdir+pathseperator)

            #create a full pathname to the current image
            imagefilepath = os.path.join(imagesubdir, imagefile)

            #read the numpy-array formated image into the loaded_image variable
            loaded_image = cv2.imread(imagefilepath, cv2.IMREAD_COLOR)
            
            #make sure the cv2.imread actual returned an image
            if type(loaded_image) != type(None):
                
                # resize the image to so all are the same size
                resized_img = cv2.resize(loaded_image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LANCZOS4)
                #add images and labels to an array
                images.append(resized_img)
                labels.append(catagoryNumber)
   # labelsset = set(labels)
    #print(len(labelsset))
    
    return (images, labels)



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    model = tf.keras.Sequential()

    #create a layer that will look for the very small features
    model.add(tf.keras.layers.Conv2D(16, (2, 2), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ))
    
    #create a layer that will look for the medium structures like lines, corners, archs, circles that are created by the connected smallest features
    model.add(tf.keras.layers.Conv2D(32, (4, 4), activation="relu"))

   #look for large feature structures like edges, structures and their relationship, patterns  
    model.add(tf.keras.layers.Conv2D(64, (8, 8), activation="relu"))
    
    #shrink the size of the convolution images by sampling
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    #process the image into a 1 dimensional structure which is needed to add to the fully connected neural network
    model.add(tf.keras.layers.Flatten())

    # This is a Hidden/Dense layer will feed all outputs from the flattened layer to all its neurons.
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # BatchNormalization Ensures all all data is scaled so their ranges are within 0 and 1
    # This is necessary because during each epoc during sthocastic Gradient Decent some features one or more of the weights of the neurons become much larger than other neurons 
    # and the increse will cascade to the next Dense layer and cause instability.
    # This is needed between the 2 Dense layers that I am using. Without this line of code the accuracy of the model is greatly lowered
    model.add(tf.keras.layers.BatchNormalization())

    #prevent over fitting of the data by randomly selecting neural net nodes to be dropped-out
    model.add(tf.keras.layers.Dropout(rate=0.50))

    # Add an output layer with output units for all NUM_CATEGORIES
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))
    
    # I used trial an error to determine the best setting to compile the model with
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["CategoricalAccuracy"]
    )

    return model

if __name__ == "__main__":
    main()
