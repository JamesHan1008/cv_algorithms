import argparse
import json
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from .common.model_evaluation import get_confusion_matrix
from .common.model_util import load_trained_model

DATA_SET = "fruits_360"
dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))
model_path = f"{root_path}/algorithms/resources/resnet/models/{DATA_SET}.h5"
model = None

with open(f"{root_path}/algorithms/resources/resnet/models/{DATA_SET}_label_map.json") as f:
    label_map = json.load(f)


def load_model():
    global model
    model = load_trained_model(model_path)
    if model is None:
        raise Exception("failed to load model")
    return model


def train(debug: bool, epochs: int, model: Model = None):
    image_size = [100, 100]
    batch_size = 32

    # https://www.kaggle.com/moltean/fruits
    train_path = f"{root_path}/images/fruits-360-small/Training"
    valid_path = f"{root_path}/images/fruits-360-small/Validation"

    num_classes = len(glob(train_path + "/*"))
    num_train = len(glob(train_path + "/*/*.jp*g"))
    num_valid = len(glob(valid_path + "/*/*.jp*g"))

    if debug:
        # show a regular image
        image_files = glob(train_path + "/*/*.jp*g")
        plt.imshow(image.load_img(np.random.choice(image_files)))
        plt.show()

    # add preprocessing layer to the front of VGG
    # if this line errors out, download this file and place it in ~/.keras/models/ :
    # resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    res = ResNet50(input_shape=image_size + [3], weights="imagenet", include_top=False)

    # don't train existing weights
    for layer in res.layers:
        layer.trainable = False

    # new layers
    x = Flatten()(res.output)
    # x = Dense(1000, activation="relu")(x)
    prediction = Dense(num_classes, activation="softmax")(x)

    # create a new model if a pre-trained one is not provided
    if model is None:
        model = Model(inputs=res.input, outputs=prediction)

    if debug:
        # show a summary of the model's layers
        model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )

    # used to generate new transfer images to help with generalization
    gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,  # color transform to match Caffe (the library VGG is built in)
    )

    # test out the generator
    test_gen = gen.flow_from_directory(valid_path, target_size=image_size)
    print(test_gen.class_indices)
    labels = [None] * len(test_gen.class_indices)
    for k, v in test_gen.class_indices.items():
        labels[v] = k

    if debug:
        # show a strangely colored image (due to VGG weights being BGR)
        for x, y in test_gen:
            print("min:", x[0].min(), "max:", x[0].max())
            plt.title(labels[np.argmax(y[0])])
            plt.imshow(x[0])
            plt.show()
            break

    # create generators
    train_generator = gen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
    )
    valid_generator = gen.flow_from_directory(
        valid_path,
        target_size=image_size,
        batch_size=batch_size,
    )

    # fit the model
    r = model.fit_generator(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        steps_per_epoch=num_train // batch_size,
        validation_steps=num_valid // batch_size,
    )

    if debug:
        print(r.history)
        print(r.history["loss"])
        print(r.history["val_loss"])
        print(r.history["acc"])
        print(r.history["val_acc"])

        # plot loss
        plt.plot(r.history["loss"], label="train loss")
        plt.plot(r.history["val_loss"], label="val loss")
        plt.legend()
        plt.show()

        # plot accuracies
        plt.plot(r.history["acc"], label="train acc")
        plt.plot(r.history["val_acc"], label="val acc")
        plt.legend()
        plt.show()

        # show confusion matrices
        train_cm = get_confusion_matrix(
            data_path=train_path,
            num_images=num_train,
            image_size=image_size,
            batch_size=batch_size,
            gen=gen,
            model=model,
        )
        print(train_cm)
        valid_cm = get_confusion_matrix(
            data_path=valid_path,
            num_images=num_valid,
            image_size=image_size,
            batch_size=batch_size,
            gen=gen,
            model=model,
        )
        print(valid_cm)

    model.save(model_path)


def classify(X) -> str:
    """
    Classify an image
    :param X: 3-D array of size: W x L x C
    :return: a label
    """
    if model is None:
        load_model()

    # model takes in a 4-D array where the first dimension is the number of images
    prediction = model.predict(X.reshape(1, 100, 100, 3))

    label = str(np.argmax(prediction, axis=1)[0])

    return label_map[label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-l", "--load", action="store_true", help="Load existing model")
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging mode")
    args = vars(parser.parse_args())

    if args["load"]:
        model = load_model()
    else:
        model = None

    if args["train"]:
        epochs = args.get("epochs", 1)
        train(
            debug=args["debug"],
            epochs=epochs,
            model=model,
        )


if __name__ == "__main__":
    main()
